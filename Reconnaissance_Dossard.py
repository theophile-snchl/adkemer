import streamlit as st
import zipfile
import io
import cv2
import easyocr
import numpy as np
import pandas as pd
from collections import Counter
import itertools
from concurrent.futures import ThreadPoolExecutor

# Dictionnaire des traductions
TRANSLATIONS = {
    "fr": {
        "title": "🏃‍♂️ Adkemer - An diskenn klask",
        "upload_label": "Uploader un fichier ZIP contenant des photos JPG",
        "confidence_threshold": "Seuil de confiance minimum pour la reconnaissance des dossards",
        "min_bib_length": "Longueur minimale du dossard (nombre de chiffres sur les dossards)",
        "search_heading": "Numéros de dossard à rechercher",
        "add_bib": "➕ Ajouter un numéro de dossard",
        "remove_bib": "➖ Retirer un numéro de dossard",
        "search_button": "🔍 Lancer la recherche",
        "no_zip_warning": "Merci d'uploader un fichier ZIP contenant les photos.",
        "no_bib_warning": "Merci d’entrer au moins un numéro de dossard.",
        "processing_images": "Traitement des images...",
        "images_processed": "**🖼️ {}/{} image(s) traitée(s)**",
        "no_match": "Aucune photo trouvée pour les dossards spécifiés.",
        "found_match": "{} photo(s) trouvée(s) contenant les dossards spécifiés.",
        "preview_title": "### Aperçu des photos détectées",
        "download_button": "📦 Télécharger le ZIP des photos trouvées",
        "footer": "💻 Développé par <strong>Théophile Sénéchal</strong> • 2025",
        "show_stats": "Afficher les statistiques après la recherche"
    },
    "en": {
        "title": "🏃‍♂️ Adkemer - An diskenn klask",
        "upload_label": "Upload a ZIP file containing JPG photos",
        "confidence_threshold": "Minimum confidence threshold for bib number recognition",
        "min_bib_length": "Minimum bib number length (number of digits on bibs)",
        "search_heading": "Bib numbers to search for",
        "add_bib": "➕ Add a bib number",
        "remove_bib": "➖ Remove a bib number",
        "search_button": "🔍 Start search",
        "no_zip_warning": "Please upload a ZIP file with photos.",
        "no_bib_warning": "Please enter at least one bib number.",
        "processing_images": "Processing images...",
        "images_processed": "**🖼️ {}/{} image(s) processed**",
        "no_match": "No photos found for the specified bib numbers.",
        "found_match": "{} photo(s) found containing the specified bib numbers.",
        "preview_title": "### Preview of detected photos",
        "download_button": "📦 Download ZIP of found photos",
        "footer": "💻 Developed by <strong>Théophile Sénéchal</strong> • 2025",
        "show_stats": "Show statistics after search"
    },
    "es": {
        "title": "🏃‍♂️ Adkemer - An diskenn klask",
        "upload_label": "Subir un archivo ZIP con fotos JPG",
        "confidence_threshold": "Umbral mínimo de confianza para reconocimiento de dorsales",
        "min_bib_length": "Longitud mínima del dorsal (número de cifras en los dorsales)",
        "search_heading": "Números de dorsal a buscar",
        "add_bib": "➕ Añadir un número de dorsal",
        "remove_bib": "➖ Eliminar un número de dorsal",
        "search_button": "🔍 Iniciar búsqueda",
        "no_zip_warning": "Por favor, sube un archivo ZIP con fotos.",
        "no_bib_warning": "Por favor, ingresa al menos un número de dorsal.",
        "processing_images": "Procesando imágenes...",
        "images_processed": "**🖼️ {}/{} imagen(es) procesada(s)**",
        "no_match": "No se encontraron fotos con los dorsales indicados.",
        "found_match": "{} foto(s) encontrada(s) con los dorsales indicados.",
        "preview_title": "### Vista previa de fotos detectadas",
        "download_button": "📦 Descargar ZIP con fotos encontradas",
        "footer": "💻 Desarrollado por <strong>Théophile Sénéchal</strong> • 2025",
        "show_stats": "Mostrar estadísticas después de la búsqueda"
    },
    "de": {
        "title": "🏃‍♂️ Adkemer - An diskenn klask",
        "upload_label": "ZIP-Datei mit JPG-Fotos hochladen",
        "confidence_threshold": "Minimaler Vertrauensschwellenwert für die Startnummernerkennung",
        "min_bib_length": "Minimale Länge der Startnummer (Anzahl der Ziffern auf den Startnummern)", 
        "search_heading": "Zu suchende Startnummern",
        "add_bib": "➕ Startnummer hinzufügen",
        "remove_bib": "➖ Startnummer entfernen",
        "search_button": "🔍 Suche starten",
        "no_zip_warning": "Bitte lade eine ZIP-Datei mit Fotos hoch.",
        "no_bib_warning": "Bitte mindestens eine Startnummer eingeben.",
        "processing_images": "Bilder werden verarbeitet...",
        "images_processed": "**🖼️ {}/{} Bild(er) verarbeitet**",
        "no_match": "Keine Fotos mit den angegebenen Startnummern gefunden.",
        "found_match": "{} Foto(s) mit den angegebenen Startnummern gefunden.",
        "preview_title": "### Vorschau der erkannten Fotos",
        "download_button": "📦 ZIP der gefundenen Fotos herunterladen",
        "footer": "💻 Entwickelt von <strong>Théophile Sénéchal</strong> • 2025",
        "show_stats": "Statistiken nach der Suche anzeigen"
    },
    "it": {
        "title": "🏃‍♂️ Adkemer - An diskenn klask",
        "upload_label": "Carica un file ZIP contenente foto JPG",
        "confidence_threshold": "Soglia minima di confidenza per il riconoscimento del numero di pettorale",
        "min_bib_length": "Lunghezza minima del pettorale (numero di cifre sul pettorale)",
        "search_heading": "Numeri di pettorale da cercare",
        "add_bib": "➕ Aggiungi un numero di pettorale",
        "remove_bib": "➖ Rimuovi un numero di pettorale",
        "search_button": "🔍 Avvia ricerca",
        "no_zip_warning": "Per favore carica un file ZIP con le foto.",
        "no_bib_warning": "Per favore inserisci almeno un numero di pettorale.",
        "processing_images": "Elaborazione immagini...",
        "images_processed": "**🖼️ {}/{} immagine/i elaborate**",
        "no_match": "Nessuna foto trovata con i numeri di pettorale specificati.",
        "found_match": "{} foto trovate contenenti i numeri di pettorale specificati.",
        "preview_title": "### Anteprima delle foto rilevate",
        "download_button": "📦 Scarica ZIP delle foto trovate",
        "footer": "💻 Sviluppato da <strong>Théophile Sénéchal</strong> • 2025",
        "show_stats": "Mostra le statistiche dopo la ricerca"
    },
    "pt": {
    "title": "🏃‍♂️ Adkemer - An diskenn klask",
    "upload_label": "Enviar um arquivo ZIP com fotos JPG",
    "confidence_threshold": "Limite mínimo de confiança para reconhecimento de dorsais",
    "min_bib_length": "Comprimento mínimo do dorsal (número de dígitos no dorsal)",
    "search_heading": "Números de dorsal a procurar",
    "add_bib": "➕ Adicionar um número de dorsal",
    "remove_bib": "➖ Remover um número de dorsal",
    "search_button": "🔍 Iniciar busca",
    "no_zip_warning": "Por favor, envie um arquivo ZIP com as fotos.",
    "no_bib_warning": "Por favor, insira pelo menos um número de dorsal.",
    "processing_images": "Processando imagens...",
    "images_processed": "**🖼️ {}/{} imagem(ns) processada(s)**",
    "no_match": "Nenhuma foto encontrada com os dorsais especificados.",
    "found_match": "{} foto(s) encontrada(s) com os dorsais especificados.",
    "preview_title": "### Pré-visualização das fotos detectadas",
    "download_button": "📦 Baixar ZIP com as fotos encontradas",
    "footer": "💻 Desenvolvido por <strong>Théophile Sénéchal</strong> • 2025",
    "show_stats": "Mostrar estatísticas após a busca"
}

}


def renommer_photos_in_memory(zip_in):
    jpg_files = sorted([f for f in zip_in.namelist() if f.lower().endswith('.jpg')])
    return [(old_name, f"photo_{i}.jpg") for i, old_name in enumerate(jpg_files, 1)]

def traiter_image(entry):
    old_name, new_name, zip_bytes, seuil_confiance = entry
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_in:
        with zip_in.open(old_name) as file:
            image_bytes = file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    max_dim = 1000
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    results = reader.readtext(image)
    dossards_info = []
    for bbox, text, prob in results:
        if prob >= seuil_confiance:
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                dossards_info.append({"text": digits, "bbox": bbox, "conf": prob})
    dossards = [d["text"] for d in dossards_info]
    return {
        'photo': new_name,
        'dossards': dossards,
        'image_bytes': image_bytes,
        'bboxes': dossards_info
    }

def main():
    st.set_page_config(page_title="Adkemer", page_icon="🏃‍♂️", layout="centered")
    lang = st.sidebar.selectbox("🌐 Langue / Language", ["Français", "English", "Español", "Deutsch", "Italiano", "Português"])
    lang_code = {"Français": "fr", "English": "en", "Español": "es", "Deutsch": "de", "Italiano": "it", "Português": "pt"}.get(lang, "fr")
    T = TRANSLATIONS[lang_code]
    st.title(T["title"])
    seuil_confiance = st.slider(T["confidence_threshold"], 0.0, 1.0, 0.95, 0.01)
    min_dossard_length = st.slider(T["min_bib_length"], min_value=1, max_value=6, value=4)

    uploaded_zip = st.file_uploader(T["upload_label"], type=["zip"])
    st.markdown(f"### {T['search_heading']}")

    nb_dossards = st.session_state.get('nb_dossards', 1)
    cols_btn = st.columns([1, 1])
    with cols_btn[0]:
        if st.button(T["add_bib"]):
            nb_dossards += 1
            st.session_state.nb_dossards = nb_dossards
    with cols_btn[1]:
        if nb_dossards > 1 and st.button(T["remove_bib"]):
            nb_dossards -= 1
            st.session_state.nb_dossards = nb_dossards

    dossards_recherches = []
    for i in range(nb_dossards):
        value = st.text_input(f"{T['search_heading']} {i+1}", key=f"dossard_{i}")
        if value.strip():
            dossards_recherches.append(value.strip())

    afficher_stats = st.checkbox(T["show_stats"], value=True)

    if st.button(T["search_button"]):
        if uploaded_zip is None:
            st.warning(T["no_zip_warning"])
            return
        if not dossards_recherches:
            st.warning(T["no_bib_warning"])
            return

        global reader
        reader = easyocr.Reader(['fr'], gpu=False)

        zip_bytes = uploaded_zip.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_in:
            mapping = renommer_photos_in_memory(zip_in)
            entries = [(old, new, zip_bytes, seuil_confiance) for old, new in mapping]

        total = len(entries)
        progress_bar = st.progress(0)
        status = st.empty()
        status_count = st.empty()

        data = []
        with ThreadPoolExecutor() as executor:
            for i, result in enumerate(executor.map(traiter_image, entries), start=1):
                data.append(result)
                progress_bar.progress(i / total)
                status.text(f"{T['processing_images']} {int((i / total) * 100)}%")
                status_count.markdown(T["images_processed"].format(i, total))

        df = pd.DataFrame(data)

        def contient_dossard(dossards_list):
            return any(dossard in dossards_list for dossard in dossards_recherches)

        photos_trouvees = df[df['dossards'].apply(contient_dossard)]

        if afficher_stats:
            tous_dossards = list(itertools.chain.from_iterable(df['dossards']))
            dossards_filtrés = [d for d in tous_dossards if len(d) >= min_dossard_length]
            compteur_dossards = Counter(dossards_filtrés)
            top_df = pd.DataFrame(compteur_dossards.items(), columns=["Dossard", "Nombre de photos"])
            top_df = top_df.sort_values("Nombre de photos", ascending=False).head(10).reset_index(drop=True)
            st.markdown("### 📈 Top 10 des dossards les plus détectés")
            st.dataframe(top_df, use_container_width=True, hide_index=True)
            nb_sans_dossard = df[df['dossards'].apply(len) == 0].shape[0]
            st.info(f"📸 Nombre de photos sans aucun dossard détecté : **{nb_sans_dossard}**")

        rows = []
        for _, row in df.iterrows():
            for d in row['bboxes']:
                rows.append({
                    "Dossard": d["text"],
                    "Fichier": row["photo"],
                    "Confiance": round(d["conf"], 4)
                })
        excel_df = pd.DataFrame(rows)
        excel_df_counts = excel_df.groupby("Dossard").agg(
            Nombre_photos=("Fichier", "nunique"),
            Confiance_moyenne=("Confiance", "mean"),
            Fichiers=("Fichier", lambda x: ", ".join(sorted(set(x))))
        ).reset_index()

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            excel_df.to_excel(writer, index=False, sheet_name="Dossards_détaillés")
            excel_df_counts.to_excel(writer, index=False, sheet_name="Résumé_par_dossard")
        excel_buffer.seek(0)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_out:
            for _, row in photos_trouvees.iterrows():
                zip_out.writestr(row['photo'], row['image_bytes'])
        zip_buffer.seek(0)

        st.session_state.df = df
        st.session_state.photos_trouvees = photos_trouvees
        st.session_state.excel_buffer = excel_buffer.getvalue()
        st.session_state.zip_buffer = zip_buffer.getvalue()
        st.session_state.dossards_recherches = dossards_recherches

    if "df" in st.session_state and "photos_trouvees" in st.session_state:
        df = st.session_state.df
        photos_trouvees = st.session_state.photos_trouvees
        dossards_recherches = st.session_state.get("dossards_recherches", [])
        excel_buffer = io.BytesIO(st.session_state.excel_buffer)
        zip_buffer = io.BytesIO(st.session_state.zip_buffer)

        st.download_button(
            label="📊 Télécharger le fichier Excel des dossards",
            data=excel_buffer,
            file_name="dossards_photos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if photos_trouvees.empty:
            st.warning(T["no_match"])
        else:
            st.success(T["found_match"].format(len(photos_trouvees)))
            st.markdown(T["preview_title"])
            cols = st.columns(min(5, len(photos_trouvees)))
            for i, (_, row) in enumerate(photos_trouvees.iterrows()):
                img_array = np.frombuffer(row['image_bytes'], np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                for dossard_info in row['bboxes']:
                    if dossard_info["text"] in dossards_recherches:
                        (tl, tr, br, bl) = dossard_info["bbox"]
                        pts = np.array([tl, tr, br, bl], dtype=np.int32)
                        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with cols[i % 5]:
                    st.image(img, use_container_width=True)

            st.download_button(
                label=T["download_button"],
                data=zip_buffer,
                file_name="photos_dossards.zip",
                mime="application/zip"
            )

    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>{T['footer']}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()