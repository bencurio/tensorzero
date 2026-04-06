use super::check_column_exists;
use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

/// This migration adds an `api_key_public_id` column to `ChatInference`,
/// `JsonInference`, and `ModelInference` so that per-key usage queries
/// can be answered without scanning the tags JSON blob.
pub struct Migration0053<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0053";

#[async_trait]
impl Migration for Migration0053<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        for table in ["ChatInference", "JsonInference", "ModelInference"] {
            if !check_table_exists(self.clickhouse, table, MIGRATION_ID).await? {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: format!("{table} table does not exist"),
                }));
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        for table in ["ChatInference", "JsonInference", "ModelInference"] {
            if !check_column_exists(self.clickhouse, table, "api_key_public_id", MIGRATION_ID)
                .await?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        for table in ["ChatInference", "JsonInference", "ModelInference"] {
            self.clickhouse
                .run_query_synchronous_no_params(format!(
                    "ALTER TABLE {table}{on_cluster_name} ADD COLUMN IF NOT EXISTS api_key_public_id Nullable(String)"
                ))
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            r"
            ALTER TABLE ChatInference{on_cluster_name} DROP COLUMN api_key_public_id;
            ALTER TABLE JsonInference{on_cluster_name} DROP COLUMN api_key_public_id;
            ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN api_key_public_id;
            "
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        for table in ["ChatInference", "JsonInference", "ModelInference"] {
            if !check_column_exists(self.clickhouse, table, "api_key_public_id", MIGRATION_ID)
                .await?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
