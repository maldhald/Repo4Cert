/*
Reconcile local DynamicTable GId tables in ReadModelProcessor from postgres_fdw foreign tables.

Source foreign tables in ReadModelProcessor:
  "DynamicTable"."DynamicTable"
  "DynamicTable"."DynamicColumn"
  "DynamicTable"."DynamicCellData"

Target local physical tables in ReadModelProcessor:
  "DynamicTable"."<DynamicTable.GId>"

This version matches the DDL you provided:
  - All dynamic columns are created as character varying COLLATE pg_catalog."default".
  - "RowNumber" is bigint NOT NULL PRIMARY KEY.
  - "OrganizationId" is bigint.
  - The local target object check only accepts ordinary/partitioned tables, not foreign tables.
  - Existing non-null target cell values are not overwritten.

Recommended dry run:
  BEGIN;
  CALL "DynamicTable".reconcile_gid_tables(false, NULL, true, 'lobdev');
  SELECT * FROM pg_temp.dynamic_table_reconcile_audit ORDER BY table_id;
  ROLLBACK;

Recommended execute:
  BEGIN;
  CALL "DynamicTable".reconcile_gid_tables(true, NULL, true, 'lobdev');
  SELECT * FROM pg_temp.dynamic_table_reconcile_audit ORDER BY table_id;
  COMMIT;
*/

CREATE OR REPLACE PROCEDURE "DynamicTable".reconcile_gid_tables(
    p_execute boolean DEFAULT false,
    p_table_id bigint DEFAULT NULL,
    p_fill_null_cells boolean DEFAULT true,
    p_owner name DEFAULT NULL
)
LANGUAGE plpgsql
AS $$
DECLARE
    r_table record;
    v_target_relkind "char";
    v_target_exists boolean;
    v_create_cols text;
    v_pivot_cols text;
    v_insert_cols text;
    v_select_cols text;
    v_update_set text;
    v_update_where text;
    v_sql text;
    v_actual_rows_before bigint;
    v_expected_rows bigint;
    v_inserted_rows bigint;
    v_updated_rows bigint;
    v_duplicate_column_count bigint;
BEGIN
    CREATE TEMP TABLE IF NOT EXISTS dynamic_table_reconcile_audit (
        audit_time timestamptz NOT NULL DEFAULT clock_timestamp(),
        table_id bigint NOT NULL,
        gid uuid NOT NULL,
        target_table text NOT NULL,
        existed_before boolean NOT NULL,
        expected_rows_from_cell_data bigint NOT NULL,
        target_rows_before bigint,
        missing_rows_before bigint,
        inserted_rows bigint NOT NULL DEFAULT 0,
        updated_existing_rows bigint NOT NULL DEFAULT 0,
        duplicate_column_names bigint NOT NULL DEFAULT 0,
        executed boolean NOT NULL,
        note text
    ) ON COMMIT PRESERVE ROWS;

    TRUNCATE dynamic_table_reconcile_audit;

    FOR r_table IN
        SELECT dt."Id" AS table_id,
               dt."GId"::uuid AS gid,
               dt."OrganizationId" AS organization_id,
               dt."StatusId" AS status_id,
               dt."Name" AS table_name
        FROM "DynamicTable"."DynamicTable" dt
        WHERE dt."GId" IS NOT NULL
          AND (p_table_id IS NULL OR dt."Id" = p_table_id)
        ORDER BY dt."Id"
    LOOP
        v_inserted_rows := 0;
        v_updated_rows := 0;
        v_target_relkind := NULL;
        v_target_exists := false;

        SELECT c.relkind
        INTO v_target_relkind
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'DynamicTable'
          AND c.relname = r_table.gid::text
        LIMIT 1;

        v_target_exists := v_target_relkind IN ('r', 'p');

        SELECT count(DISTINCT dcd."RowNumber")
        INTO v_expected_rows
        FROM "DynamicTable"."DynamicCellData" dcd
        WHERE dcd."TableId" = r_table.table_id;

        IF v_target_relkind IS NOT NULL AND NOT v_target_exists THEN
            INSERT INTO dynamic_table_reconcile_audit (
                table_id, gid, target_table, existed_before,
                expected_rows_from_cell_data, target_rows_before, missing_rows_before,
                inserted_rows, updated_existing_rows, duplicate_column_names, executed, note
            ) VALUES (
                r_table.table_id,
                r_table.gid,
                format('%I.%I', 'DynamicTable', r_table.gid::text),
                false,
                v_expected_rows,
                NULL,
                NULL,
                0,
                0,
                0,
                p_execute,
                format('Skipped: object exists but is relkind %s, not an ordinary/partitioned local table.', v_target_relkind)
            );
            CONTINUE;
        END IF;

        IF v_target_exists THEN
            EXECUTE format('SELECT count(*) FROM %I.%I', 'DynamicTable', r_table.gid::text)
            INTO v_actual_rows_before;
        ELSE
            v_actual_rows_before := 0;
        END IF;

        SELECT count(*)
        INTO v_duplicate_column_count
        FROM (
            SELECT dc."Name"
            FROM "DynamicTable"."DynamicColumn" dc
            WHERE dc."TableId" = r_table.table_id
            GROUP BY dc."Name"
            HAVING count(*) > 1
        ) d;

        IF v_duplicate_column_count > 0 THEN
            INSERT INTO dynamic_table_reconcile_audit (
                table_id, gid, target_table, existed_before,
                expected_rows_from_cell_data, target_rows_before, missing_rows_before,
                inserted_rows, updated_existing_rows, duplicate_column_names, executed, note
            ) VALUES (
                r_table.table_id,
                r_table.gid,
                format('%I.%I', 'DynamicTable', r_table.gid::text),
                v_target_exists,
                v_expected_rows,
                v_actual_rows_before,
                greatest(v_expected_rows - coalesce(v_actual_rows_before, 0), 0),
                0,
                0,
                v_duplicate_column_count,
                p_execute,
                'Skipped: duplicate DynamicColumn.Name values would create duplicate target columns.'
            );
            CONTINUE;
        END IF;

        SELECT string_agg(format('%I character varying COLLATE pg_catalog."default"', dc."Name"), ', ' ORDER BY dc."Order", dc."Id")
        INTO v_create_cols
        FROM "DynamicTable"."DynamicColumn" dc
        WHERE dc."TableId" = r_table.table_id;

        IF p_execute AND NOT v_target_exists THEN
            EXECUTE format(
                'CREATE TABLE %I.%I ("RowNumber" bigint NOT NULL%s%s, "OrganizationId" bigint, CONSTRAINT %I PRIMARY KEY ("RowNumber")) TABLESPACE pg_default',
                'DynamicTable', r_table.gid::text,
                CASE WHEN coalesce(v_create_cols, '') = '' THEN '' ELSE ', ' END,
                coalesce(v_create_cols, ''),
                left(r_table.gid::text || '_pkey', 63)
            );

            IF p_owner IS NOT NULL THEN
                EXECUTE format('ALTER TABLE %I.%I OWNER TO %I', 'DynamicTable', r_table.gid::text, p_owner);
            END IF;
        END IF;

        IF p_execute THEN
            FOR v_sql IN
                SELECT format(
                    'ALTER TABLE %I.%I ADD COLUMN IF NOT EXISTS %I character varying COLLATE pg_catalog."default"',
                    'DynamicTable', r_table.gid::text,
                    dc."Name"
                )
                FROM "DynamicTable"."DynamicColumn" dc
                WHERE dc."TableId" = r_table.table_id
                ORDER BY dc."Order", dc."Id"
            LOOP
                EXECUTE v_sql;
            END LOOP;

            EXECUTE format('ALTER TABLE %I.%I ADD COLUMN IF NOT EXISTS "OrganizationId" bigint', 'DynamicTable', r_table.gid::text);
        END IF;

        SELECT string_agg(format('max(dcd."Value") FILTER (WHERE dcd."ColumnId" = %s) AS %I', dc."Id", dc."Name"), ', ' ORDER BY dc."Order", dc."Id"),
               string_agg(format('%I', dc."Name"), ', ' ORDER BY dc."Order", dc."Id"),
               string_agg(format('s.%I', dc."Name"), ', ' ORDER BY dc."Order", dc."Id"),
               string_agg(format('%1$I = COALESCE(t.%1$I, s.%1$I)', dc."Name"), ', ' ORDER BY dc."Order", dc."Id"),
               string_agg(format('(t.%1$I IS NULL AND s.%1$I IS NOT NULL)', dc."Name"), ' OR ' ORDER BY dc."Order", dc."Id")
        INTO v_pivot_cols, v_insert_cols, v_select_cols, v_update_set, v_update_where
        FROM "DynamicTable"."DynamicColumn" dc
        WHERE dc."TableId" = r_table.table_id;

        IF p_execute AND v_expected_rows > 0 THEN
            v_sql := format($fmt$
                WITH src AS (
                    SELECT dcd."RowNumber"::bigint AS "RowNumber"%s%s
                    FROM "DynamicTable"."DynamicCellData" dcd
                    WHERE dcd."TableId" = %s
                    GROUP BY dcd."RowNumber"
                )
                INSERT INTO %I.%I ("RowNumber"%s%s, "OrganizationId")
                SELECT s."RowNumber"%s%s, %s::bigint
                FROM src s
                WHERE NOT EXISTS (
                    SELECT 1 FROM %I.%I t WHERE t."RowNumber" = s."RowNumber"
                )
            $fmt$,
                CASE WHEN coalesce(v_pivot_cols, '') = '' THEN '' ELSE ', ' END,
                coalesce(v_pivot_cols, ''),
                r_table.table_id,
                'DynamicTable', r_table.gid::text,
                CASE WHEN coalesce(v_insert_cols, '') = '' THEN '' ELSE ', ' END,
                coalesce(v_insert_cols, ''),
                CASE WHEN coalesce(v_select_cols, '') = '' THEN '' ELSE ', ' END,
                coalesce(v_select_cols, ''),
                coalesce(r_table.organization_id::text, 'NULL'),
                'DynamicTable', r_table.gid::text
            );
            EXECUTE v_sql;
            GET DIAGNOSTICS v_inserted_rows = ROW_COUNT;

            IF p_fill_null_cells AND coalesce(v_update_set, '') <> '' THEN
                v_sql := format($fmt$
                    WITH src AS (
                        SELECT dcd."RowNumber"::bigint AS "RowNumber", %s
                        FROM "DynamicTable"."DynamicCellData" dcd
                        WHERE dcd."TableId" = %s
                        GROUP BY dcd."RowNumber"
                    )
                    UPDATE %I.%I t
                    SET %s,
                        "OrganizationId" = COALESCE(t."OrganizationId", %s::bigint)
                    FROM src s
                    WHERE t."RowNumber" = s."RowNumber"
                      AND (%s OR t."OrganizationId" IS NULL)
                $fmt$,
                    v_pivot_cols,
                    r_table.table_id,
                    'DynamicTable', r_table.gid::text,
                    v_update_set,
                    coalesce(r_table.organization_id::text, 'NULL'),
                    v_update_where
                );
                EXECUTE v_sql;
                GET DIAGNOSTICS v_updated_rows = ROW_COUNT;
            END IF;
        END IF;

        INSERT INTO dynamic_table_reconcile_audit (
            table_id, gid, target_table, existed_before,
            expected_rows_from_cell_data, target_rows_before, missing_rows_before,
            inserted_rows, updated_existing_rows, duplicate_column_names, executed, note
        ) VALUES (
            r_table.table_id,
            r_table.gid,
            format('%I.%I', 'DynamicTable', r_table.gid::text),
            v_target_exists,
            v_expected_rows,
            v_actual_rows_before,
            greatest(v_expected_rows - coalesce(v_actual_rows_before, 0), 0),
            v_inserted_rows,
            v_updated_rows,
            v_duplicate_column_count,
            p_execute,
            CASE
                WHEN NOT p_execute THEN 'Dry run only. No DDL/DML executed.'
                WHEN NOT v_target_exists THEN 'Target table was missing and was created.'
                ELSE 'Target table existed; missing rows and null cells were reconciled.'
            END
        );
    END LOOP;
END;
$$;

-- Pre-check 1: GId values that do not have a local ordinary or partitioned table.
SELECT dt."Id" AS table_id,
       dt."Name" AS table_name,
       dt."GId",
       dt."OrganizationId",
       c.relkind AS existing_object_relkind
FROM "DynamicTable"."DynamicTable" dt
LEFT JOIN pg_class c
    ON c.relname = dt."GId"::text
LEFT JOIN pg_namespace n
    ON n.oid = c.relnamespace
   AND n.nspname = 'DynamicTable'
WHERE dt."GId" IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM pg_class c2
      JOIN pg_namespace n2 ON n2.oid = c2.relnamespace
      WHERE n2.nspname = 'DynamicTable'
        AND c2.relname = dt."GId"::text
        AND c2.relkind IN ('r', 'p')
  )
ORDER BY dt."Id";

-- Pre-check 2: duplicate target column names. These must be resolved before repair.
SELECT dc."TableId", dc."Name", count(*) AS duplicate_count
FROM "DynamicTable"."DynamicColumn" dc
GROUP BY dc."TableId", dc."Name"
HAVING count(*) > 1
ORDER BY dc."TableId", dc."Name";

-- Pre-check 3: source cell count by table.
SELECT dt."Id" AS table_id,
       dt."GId",
       count(DISTINCT dcd."RowNumber") AS expected_rows_from_cell_data,
       count(*) AS source_cell_count
FROM "DynamicTable"."DynamicTable" dt
LEFT JOIN "DynamicTable"."DynamicCellData" dcd
    ON dcd."TableId" = dt."Id"
WHERE dt."GId" IS NOT NULL
GROUP BY dt."Id", dt."GId"
ORDER BY dt."Id";
