from sentence_transformers import SentenceTransformer

# Download and load the model once
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Save the model locally
local_path = "./models/local_all-MiniLM-L6-v2"
model.save(local_path)