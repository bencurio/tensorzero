-- Add api_key_public_id column to inference metadata tables.
-- Nullable because auth may be disabled and historical rows lack a value.
ALTER TABLE tensorzero.chat_inferences ADD COLUMN IF NOT EXISTS api_key_public_id CHAR(12);
ALTER TABLE tensorzero.json_inferences ADD COLUMN IF NOT EXISTS api_key_public_id CHAR(12);
ALTER TABLE tensorzero.model_inferences ADD COLUMN IF NOT EXISTS api_key_public_id CHAR(12);

-- Partial indexes for efficient per-key querying (inherited by partitions).
CREATE INDEX IF NOT EXISTS idx_chat_inferences_api_key
  ON tensorzero.chat_inferences(api_key_public_id) WHERE api_key_public_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_json_inferences_api_key
  ON tensorzero.json_inferences(api_key_public_id) WHERE api_key_public_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_model_inferences_api_key
  ON tensorzero.model_inferences(api_key_public_id) WHERE api_key_public_id IS NOT NULL;
