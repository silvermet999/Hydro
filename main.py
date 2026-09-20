import pandas as pd



df = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Niveaux_journaliers")
df_metadata = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Metadonnees_puits")
df_diag = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Couverture_series")

features = df.drop("DATE", axis=1)
availability = features.notna().astype(int)
numeric_cols = df.select_dtypes(include="number").columns

df_add = df_metadata[["ID_PUITS", "MUNICIPALITE"]]
# grouped = df_add.groupby('MUNICIPALITE')['ID_PUITS'].apply(list)

GROUP_MAP = {
    # 1. Nord-du-Québec (Baie-James / Nunavik)
    "Umiujaq": "1_Nord-du-Quebec",
    "Eeyou Istchee Baie-James": "1_Nord-du-Quebec",
    "Eastmain": "1_Nord-du-Quebec",
    "Whapmagoostui-Kuujjuarapik": "1_Nord-du-Quebec",

    # 2. Côte-Nord
    "Longue-Rive": "2_Cote-Nord",
    "Pointe-Lebel": "2_Cote-Nord",
    "Rivière-au-Tonnerre": "2_Cote-Nord",
    "Sept-Îles": "2_Cote-Nord",
    "Baie-Trinité": "2_Cote-Nord",
    "Natashquan": "2_Cote-Nord",
    "Fermont": "2_Cote-Nord",

    # 3. Saguenay-Lac-Saint-Jean & Charlevoix
    "Saint-Félix-d'Otis": "3_Saguenay-LSJ_Charlevoix",
    "Sainte-Rose-du-Nord": "3_Saguenay-LSJ_Charlevoix",
    "Dolbeau-Mistassini": "3_Saguenay-LSJ_Charlevoix",
    "Petit-Saguenay": "3_Saguenay-LSJ_Charlevoix",
    "Lac-Pikauba": "3_Saguenay-LSJ_Charlevoix",
    "L'Anse-Saint-Jean": "3_Saguenay-LSJ_Charlevoix",
    "Girardville": "3_Saguenay-LSJ_Charlevoix",
    "La Doré": "3_Saguenay-LSJ_Charlevoix",
    "Saint-Félicien": "3_Saguenay-LSJ_Charlevoix",
    "Métabetchouan - Lac-à-la-Croix": "3_Saguenay-LSJ_Charlevoix",
    "Sainte-Hedwidge": "3_Saguenay-LSJ_Charlevoix",
    "Saint-Ludger-de-Milot": "3_Saguenay-LSJ_Charlevoix",
    "La Malbaie": "3_Saguenay-LSJ_Charlevoix",
    "Clermont": "3_Saguenay-LSJ_Charlevoix",
    "Saint-Siméon": "3_Saguenay-LSJ_Charlevoix",

    # 4. Capitale-Nationale
    "Québec": "4_Capitale-Nationale",
    "Pont-Rouge": "4_Capitale-Nationale",
    "Saint-Raymond": "4_Capitale-Nationale",
    "Saint-Léonard-de-Portneuf": "4_Capitale-Nationale",
    "Sainte-Christine-d'Auvergne": "4_Capitale-Nationale",
    "Saint-Ferréol-les-Neiges": "4_Capitale-Nationale",
    "Lac-Jacques-Cartier": "4_Capitale-Nationale",
    "Sainte-Famille-de-l'Île-d'Orléans": "4_Capitale-Nationale",

    # 5. Chaudière-Appalaches
    "Armagh": "5_Chaudiere-Appalaches",
    "Saint-Charles-de-Bellechasse": "5_Chaudiere-Appalaches",
    "Saint-Raphaël": "5_Chaudiere-Appalaches",
    "Frampton": "5_Chaudiere-Appalaches",
    "Saint-Luc-de-Bellechasse": "5_Chaudiere-Appalaches",
    "Saint-Anselme": "5_Chaudiere-Appalaches",
    "Saint-Agapit": "5_Chaudiere-Appalaches",
    "Saint-Martin": "5_Chaudiere-Appalaches",
    "Saint-Zacharie": "5_Chaudiere-Appalaches",
    "Sainte-Justine": "5_Chaudiere-Appalaches",
    "Saint-Honoré-de-Shenley": "5_Chaudiere-Appalaches",
    "Saint-Gilles": "5_Chaudiere-Appalaches",
    "Saint-Georges": "5_Chaudiere-Appalaches",
    "East Broughton": "5_Chaudiere-Appalaches",
    "Saint-Édouard-de-Lotbinière": "5_Chaudiere-Appalaches",
    "Leclercville": "5_Chaudiere-Appalaches",
    "Saint-Jacques-de-Leeds": "5_Chaudiere-Appalaches",
    "Thetford Mines": "5_Chaudiere-Appalaches",
    "Saint-Vallier": "5_Chaudiere-Appalaches",
    "L'Islet": "5_Chaudiere-Appalaches",
    "Saint-Antoine-de-Tilly": "5_Chaudiere-Appalaches",
    "Lévis": "5_Chaudiere-Appalaches",
    "Berthier-sur-Mer": "5_Chaudiere-Appalaches",
    "Saint-Roch-des-Aulnaies": "5_Chaudiere-Appalaches",
    "Disraeli": "5_Chaudiere-Appalaches",
    "Saint-Théophile": "5_Chaudiere-Appalaches",
    "Tourville": "5_Chaudiere-Appalaches",
    "Saint-Magloire": "5_Chaudiere-Appalaches",
    "Saint-Pamphile": "5_Chaudiere-Appalaches",
    "Irlande": "5_Chaudiere-Appalaches",

    # 6. Bas-Saint-Laurent & Gaspésie-Îles-de-la-Madeleine
    "Notre-Dame-des-Neiges": "6_BasStLaurent_Gaspesie",
    "Saint-Narcisse-de-Rimouski": "6_BasStLaurent_Gaspesie",
    "Saint-Fabien": "6_BasStLaurent_Gaspesie",
    "Sainte-Luce": "6_BasStLaurent_Gaspesie",
    "Baie-des-Sables": "6_BasStLaurent_Gaspesie",
    "Matane": "6_BasStLaurent_Gaspesie",
    "Saint-René-de-Matane": "6_BasStLaurent_Gaspesie",
    "Padoue": "6_BasStLaurent_Gaspesie",
    "Sainte-Angèle-de-Mérici": "6_BasStLaurent_Gaspesie",
    "Matapédia": "6_BasStLaurent_Gaspesie",
    "Saint-Antonin": "6_BasStLaurent_Gaspesie",
    "Les Îles-de-la-Madeleine": "6_BasStLaurent_Gaspesie",
    "Paspébiac": "6_BasStLaurent_Gaspesie",
    "Nouvelle": "6_BasStLaurent_Gaspesie",
    "Lac-Huron": "6_BasStLaurent_Gaspesie",
    "Gaspé": "6_BasStLaurent_Gaspesie",
    "Marsoui": "6_BasStLaurent_Gaspesie",
    "Mont-Albert": "6_BasStLaurent_Gaspesie",
    "Cap-Chat": "6_BasStLaurent_Gaspesie",
    "Percé": "6_BasStLaurent_Gaspesie",
    "Chandler": "6_BasStLaurent_Gaspesie",

    # 7. Mauricie, Centre-du-Québec & Estrie
    "Trois-Rivières": "7_Mauricie_Centre_Estrie",
    "Batiscan": "7_Mauricie_Centre_Estrie",
    "Saint-Étienne-des-Grès": "7_Mauricie_Centre_Estrie",
    "Sainte-Angèle-de-Prémont": "7_Mauricie_Centre_Estrie",
    "Saint-Stanislas": "7_Mauricie_Centre_Estrie",
    "Villeroy": "7_Mauricie_Centre_Estrie",
    "Sainte-Marie-de-Blandford": "7_Mauricie_Centre_Estrie",
    "Daveluyville": "7_Mauricie_Centre_Estrie",
    "Manseau": "7_Mauricie_Centre_Estrie",
    "Bécancour": "7_Mauricie_Centre_Estrie",
    "Drummondville": "7_Mauricie_Centre_Estrie",
    "Saint-Guillaume": "7_Mauricie_Centre_Estrie",
    "Baie-du-Febvre": "7_Mauricie_Centre_Estrie",
    "Victoriaville": "7_Mauricie_Centre_Estrie",
    "Saint-Rémi-de-Tingwick": "7_Mauricie_Centre_Estrie",
    "Sainte-Monique": "7_Mauricie_Centre_Estrie",
    "Mont-Carmel": "7_Mauricie_Centre_Estrie",
    "Sainte-Christine": "7_Mauricie_Centre_Estrie",
    "Saint-Isidore-de-Clifton": "7_Mauricie_Centre_Estrie",
    "Sherbrooke": "7_Mauricie_Centre_Estrie",
    "Val-Joli": "7_Mauricie_Centre_Estrie",
    "Coaticook": "7_Mauricie_Centre_Estrie",
    "Compton": "7_Mauricie_Centre_Estrie",
    "Stanstead": "7_Mauricie_Centre_Estrie",
    "Magog": "7_Mauricie_Centre_Estrie",
    "Orford": "7_Mauricie_Centre_Estrie",
    "Weedon": "7_Mauricie_Centre_Estrie",
    "Ulverton": "7_Mauricie_Centre_Estrie",
    "Stornoway": "7_Mauricie_Centre_Estrie",
    "Nantes": "7_Mauricie_Centre_Estrie",
    "Notre-Dame-des-Bois": "7_Mauricie_Centre_Estrie",
    "Cookshire-Eaton": "7_Mauricie_Centre_Estrie",
    "Frelighsburg": "7_Mauricie_Centre_Estrie",
    "Potton": "7_Mauricie_Centre_Estrie",
    "Eastman": "7_Mauricie_Centre_Estrie",
    "Brome": "7_Mauricie_Centre_Estrie",
    "Valcourt": "7_Mauricie_Centre_Estrie",
    "Dudswell": "7_Mauricie_Centre_Estrie",
    "Val-des-Sources": "7_Mauricie_Centre_Estrie",
    "Saint-Camille": "7_Mauricie_Centre_Estrie",
    "Bromont": "7_Mauricie_Centre_Estrie",
    "Cowansville": "7_Mauricie_Centre_Estrie",
    "Sutton": "7_Mauricie_Centre_Estrie",

    # 8. Montérégie
    "Mercier": "8_Monteregie",
    "Sainte-Martine": "8_Monteregie",
    "Rougemont": "8_Monteregie",
    "Franklin": "8_Monteregie",
    "Godmanchester": "8_Monteregie",
    "Saint-Rémi": "8_Monteregie",
    "Saint-Jean-sur-Richelieu": "8_Monteregie",
    "Saint-Ignace-de-Stanbridge": "8_Monteregie",
    "Elgin": "8_Monteregie",
    "Saint-Michel": "8_Monteregie",
    "Sainte-Clotilde": "8_Monteregie",
    "Saint-Patrice-de-Sherrington": "8_Monteregie",
    "Saint-Anicet": "8_Monteregie",
    "Havelock": "8_Monteregie",
    "Ormstown": "8_Monteregie",
    "Saint-Urbain-Premier": "8_Monteregie",
    "Saint-Isidore": "8_Monteregie",
    "Saint-Lazare": "8_Monteregie",
    "Saint-Télesphore": "8_Monteregie",
    "Sainte-Marthe": "8_Monteregie",
    "Calixa-Lavallée": "8_Monteregie",
    "Saint-Amable": "8_Monteregie",
    "Saint-Mathias-sur-Richelieu": "8_Monteregie",
    "Saint-Ours": "8_Monteregie",
    "Sainte-Victoire-de-Sorel": "8_Monteregie",
    "Très-Saint-Sacrement": "8_Monteregie",
    "Saint-Paul-d'Abbotsford": "8_Monteregie",
    "Saint-Alphonse-de-Granby": "8_Monteregie",
    "Saint-Marcel-de-Richelieu": "8_Monteregie",
    "Saint-Hugues": "8_Monteregie",
    "Saint-Simon": "8_Monteregie",
    "Saint-Théodore-d'Acton": "8_Monteregie",
    "Saint-Damase": "8_Monteregie",
    "Saint-Valérien-de-Milton": "8_Monteregie",
    "Saint-Hyacinthe": "8_Monteregie",
    "Sainte-Angèle-de-Monnoir": "8_Monteregie",

    # 9. Outaouais & Abitibi-Témiscamingue
    "Gatineau": "9_Outaouais_Abitibi",
    "Gracefield": "9_Outaouais_Abitibi",
    "Messines": "9_Outaouais_Abitibi",
    "Cantley": "9_Outaouais_Abitibi",
    "L'Isle-aux-Allumettes": "9_Outaouais_Abitibi",
    "Clarendon": "9_Outaouais_Abitibi",
    "Saint-André-Avellin": "9_Outaouais_Abitibi",
    "Landrienne": "9_Outaouais_Abitibi",
    "Rouyn-Noranda": "9_Outaouais_Abitibi",
    "Val-d'Or": "9_Outaouais_Abitibi",
    "Nédélec": "9_Outaouais_Abitibi",
    "La Sarre": "9_Outaouais_Abitibi",
    "Dupuy": "9_Outaouais_Abitibi",
    "La Motte": "9_Outaouais_Abitibi",
    "Barraute": "9_Outaouais_Abitibi",
    "Senneterre": "9_Outaouais_Abitibi",
    "Sainte-Gertrude-Manneville": "9_Outaouais_Abitibi",
    "Saint-Dominique-du-Rosaire": "9_Outaouais_Abitibi",
    "Témiscaming": "9_Outaouais_Abitibi",

    # 10. Laurentides & Lanaudière
    "Lachute": "10_Laurentides_Lanaudiere",
    "Brownsburg-Chatham": "10_Laurentides_Lanaudiere",
    "Arundel": "10_Laurentides_Lanaudiere",
    "La Conception": "10_Laurentides_Lanaudiere",
    "L'Ascension": "10_Laurentides_Lanaudiere",
    "Sainte-Agathe-des-Monts": "10_Laurentides_Lanaudiere",
    "Lac-des-Écorces": "10_Laurentides_Lanaudiere",
    "Morin-Heights": "10_Laurentides_Lanaudiere",
    "Piedmont": "10_Laurentides_Lanaudiere",
    "Saint-Colomban": "10_Laurentides_Lanaudiere",
    "Oka": "10_Laurentides_Lanaudiere",
    "Mirabel": "10_Laurentides_Lanaudiere",
    "Lac-Saguay": "10_Laurentides_Lanaudiere",
    "L'Ascension-de-Notre-Seigneur": "10_Laurentides_Lanaudiere",
    "Ferme-Neuve": "10_Laurentides_Lanaudiere",
    "Sainte-Anne-des-Plaines": "10_Laurentides_Lanaudiere",
    "Notre-Dame-du-Laus": "10_Laurentides_Lanaudiere",
    "Grenville-sur-la-Rouge": "10_Laurentides_Lanaudiere",
    "Mascouche": "10_Laurentides_Lanaudiere",
    "Sainte-Julienne": "10_Laurentides_Lanaudiere",
    "Saint-Michel-des-Saints": "10_Laurentides_Lanaudiere",
    "Mandeville": "10_Laurentides_Lanaudiere",
    "Saint-Calixte": "10_Laurentides_Lanaudiere",
    "Sainte-Marcelline-de-Kildare": "10_Laurentides_Lanaudiere",

    # Resolved on second pass (originally unmatched)
    "Sainte-Jeanne-d'Arc": "3_Saguenay-LSJ_Charlevoix",
    "Saint-Arsène": "6_BasStLaurent_Gaspesie",
    "Saint-Albert": "7_Mauricie_Centre_Estrie",
    "Saint-Paul-de-l'Île-aux-Noix": "8_Monteregie",
}

df_add['GROUP'] = df_add['MUNICIPALITE'].map(GROUP_MAP)

df_1 = df_add[df_add['GROUP'] == '1_Nord-du-Quebec']
df_2 = df_add[df_add['GROUP'] == '2_Cote-Nord']
df_3 = df_add[df_add['GROUP'] == '3_Saguenay-LSJ_Charlevoix']
df_4 = df_add[df_add['GROUP'] == '4_Capitale-Nationale']
df_5 = df_add[df_add['GROUP'] == '5_Chaudiere-Appalaches']
df_6 = df_add[df_add['GROUP'] == '6_BasStLaurent_Gaspesie']
df_7 = df_add[df_add['GROUP'] == '7_Mauricie_Centre_Estrie']
df_8 = df_add[df_add['GROUP'] == '8_Monteregie']
df_9 = df_add[df_add['GROUP'] == '9_Outaouais_Abitibi']
df_10 = df_add[df_add['GROUP'] == '10_Laurentides_Lanaudiere']





