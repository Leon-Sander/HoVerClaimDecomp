import sqlite3


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return conn, c

db_path = '../data/wiki_wo_links.db'
conn, cursor = connect_to_db(db_path)

dataset_type = "dev"
#data = load_obj(f"hover_dev_300_claims.json")
data = load_obj("data/mistral_retrieval_output_dev_100_likedecomposed.json")
enriched_claims = []
for item in tqdm(data):
    retrieved = item["retrieved"][:5]
    context = ""
    for title in retrieved:
        cursor.execute("SELECT * FROM documents WHERE id=?", (title,))
        output = cursor.fetchone()
        context += f"TITLE: '{output[0]}', DOCUMENT: '{output[1]}',"
    enriched_claim = chain.invoke({"claim": item["claim"], "context" : context})
    copied_item = item.copy()
    copied_item["enriched_claim"] = enriched_claim
    enriched_claims.append(copied_item)

save_obj(enriched_claims, "enriched_claims2_300.json")

"""batch_size = 8  # Define your batch size
batch_claims = []

for item in tqdm(data):
    batch_claims.append({"claim": item["claim"]})

    if len(batch_claims) == batch_size:
        outputs = chain.batch(batch_claims)
        print(outputs)
        break
        for output, batch_item in zip(outputs, batch_claims):
            copied_item = batch_item.copy()
            copied_item["decomposed_claims"] = output["output"]  # Adjust according to the actual output structure
            decomposed.append(copied_item)
        
        # Reset the batch
        batch_claims = []

# Process any remaining items in the batch
if batch_claims:
    outputs = chain.batch(batch_claims)
    for output, batch_item in zip(outputs, batch_claims):
        copied_item = batch_item.copy()
        copied_item["decomposed_claims"] = output["output"]  # Adjust according to the actual output structure
        decomposed.append(copied_item)

save_obj(decomposed, "decomposed_300_claims_few_show.json")"""