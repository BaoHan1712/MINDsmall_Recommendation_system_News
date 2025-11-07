import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1️⃣ Đọc dữ liệu cơ bản ==========
behaviors = pd.read_csv(r"MINDsmall_dev\behaviors.tsv", sep="\t",
                        names=["impression_id", "user_id", "time", "history", "impressions"])
news = pd.read_csv(r"MINDsmall_dev\news.tsv", sep="\t",
                   names=["news_id", "category", "subcategory", "title", "abstract",
                          "url", "title_entities", "abstract_entities"])

# ========== 2️⃣ Đọc embedding ==========
def load_embedding(path):
    emb = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            emb[parts[0]] = np.array(list(map(float, parts[1:])))
    return emb

entity_emb = load_embedding(r"MINDsmall_dev\entity_embedding.vec")
relation_emb = load_embedding(r"MINDsmall_dev\relation_embedding.vec")  # (chưa dùng)

# ========== 3️⃣ Gắn embedding cho từng tin tức ==========
def get_news_vector(row):
    vecs = []
    for col in ["title_entities", "abstract_entities"]:
        try:
            entities = eval(row[col])
            for ent in entities:
                wikidata_id = ent.get("WikidataId")
                if wikidata_id in entity_emb:
                    vecs.append(entity_emb[wikidata_id])
        except:
            pass
    if not vecs:
        return np.zeros(100)
    return np.mean(vecs, axis=0)

news["vector"] = news.apply(get_news_vector, axis=1)

# ========== 4️⃣ Xây dựng hồ sơ người dùng ==========
def build_user_profile(user_id):
    user_histories = behaviors[behaviors["user_id"] == user_id]["history"]
    clicked_news_ids = []
    for h in user_histories:
        if isinstance(h, str):
            clicked_news_ids.extend(h.split())
    clicked_vecs = [news.loc[news["news_id"] == nid, "vector"].values[0]
                    for nid in clicked_news_ids if nid in news["news_id"].values]
    if not clicked_vecs:
        return np.zeros(100)
    return np.mean(clicked_vecs, axis=0)

# ========== 5️⃣ Hàm đề xuất ==========
def recommend_news(user_id, top_k=5):
    user_vec = build_user_profile(user_id)
    if np.all(user_vec == 0):
        print(f"⚠️  Không có lịch sử đọc tin cho user {user_id}")
        return None
    all_news_vecs = np.stack(news["vector"].values)
    sims = cosine_similarity([user_vec], all_news_vecs)[0]
    news["score"] = sims
    top_news = news.sort_values(by="score", ascending=False).head(top_k)
    return top_news[["news_id", "title", "score"]]

# ========== 6️⃣ Thử nghiệm ==========
user_id_test = behaviors["user_id"].iloc[1]
recommendations = recommend_news(user_id_test, top_k=5)

# ========== 7️⃣ In kết quả đẹp ==========
print("\n" + "═" * 80)
print(f"🔍 GỢI Ý TIN TỨC CHO NGƯỜI DÙNG: {user_id_test}")
print("═" * 80)
for i, row in recommendations.iterrows():
    print(f" {i - recommendations.index[0] + 1:>2}. 📰  {row['title']}")
    print(f"     ➤ ID: {row['news_id']}   |   💡 Score: {row['score']:.4f}")
    print("-" * 80)
print("✅ Hoàn thành gợi ý!\n")
