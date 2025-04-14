# fungsi untuk sistem rekomendasi destinasi wisata menggunakan metode content based filtering
def travel_recommendations(inputDescription, inputCity, inputCategory, top_n=3):
    # preprocessing data input deskripsi wisata
    input_pre_desc = TextPreprocessing(inputDescription)
    input_tfidf_desc = tags_tfidf.transform([input_pre_desc])
    cosim = cosine_similarity(input_tfidf_desc, tags_matrix)[0]
    
    # filtering data berdasarkan kota dan kategori destinasi wisata
    filtered_indicies = [
        i for i in range(len(df['Place_Name']))
        if df['Category'][i] == inputCategory and
        df['City'][i] == inputCity
    ]
    
    # mengecek apakah terdapat destinasi wisata yang sesuai dengan kota dan kategori
    if not filtered_indicies:
        return []
    filtered_similarity = cosim[filtered_indicies] # mengambil nilai kemiripan data yang sesuai dengan filter
    
    # mengurutkan hasil 'top_n' teratas dari data termirip berdasarkan probabilitas tertinggi    
    sorted_order = np.argsort(filtered_similarity)[::-1][:top_n]
    recommendation = [filtered_indicies[i] for i in sorted_order]
    prob = (cosim / np.sum(cosim))*100 # mencari nilai probabilitas setiap hasil rekomendasi
    
    # menyimpan nama, kota, dan kategori destinasi wisata yang direkomendasikan
    recomendation_dict = {"Name": [], "City": [], "Category": [], "Probability": []}
    for i in recommendation:
        recomendation_dict["Name"].append(df['Place_Name'][i])
        recomendation_dict["Category"].append(df['Category'][i])
        recomendation_dict["City"].append(df['City'][i])
        recomendation_dict["Probability"].append(round(prob[i], 2))
    return recomendation_dict


# membuat kamus data untuk category dan city
category_dict = {i: cat for i, cat in enumerate(sorted(df['Category'].unique().tolist()))}
city_dict = {i: cit for i, cit in enumerate(sorted(df['City'].unique().tolist()))}

# fungsi untuk dekode data category
def decode_category(idx):
    result = [values for key, values in category_dict.items() if idx == key][0]
    return result

# fungsi untuk dekode data city
def decode_city(idx):
    result = [values for key, values in city_dict.items() if idx == key][0]
    return result


# interface sistem rekomendasi destinasi wisata
def view_result():
    # menampilkan list category dan city yang tersedia
    print("="*50)
    print(" Tourist Destination Reccomendation ".center(50, " "))
    print("="*50)
    print("List City:")
    for i, cit in city_dict.items():   
        print(f"{i+1}.".ljust(3, " ")+f"{cit}")
    print("-"*50)
    print("List Category:")
    for i, cat in category_dict.items():
        print(f"{i+1}.".ljust(3, " ")+f"{cat}")
    print("_"*50)

    # input user
    city = int(input("Input City name   : ")) # input berupa nomor urut dari list nama kota
    catg = int(input("Input Category    : ")) # input berupa nomor urut dari list kategori
    desc = str(input("Input Description : "))
    input_city = decode_city(city - 1)
    input_catg = decode_category(catg - 1)
    print("="*50)

    # hasil
    results = travel_recommendations(inputDescription=desc, inputCategory=input_catg, inputCity=input_city)
    print("Result:\n")
    for i in range(len(results['Name'])):
        print(f"{i+1}.".ljust(3, " ") + f"Name        : {results["Name"][i]}")
        print(" "*3 + f"City        : {results["City"][i]}")
        print(" "*3 + f"Category    : {results["Category"][i]}")
        print(" "*3 + f"Probability : {results["Probability"][i]}")
        print()
    print("="*50)

view_result()