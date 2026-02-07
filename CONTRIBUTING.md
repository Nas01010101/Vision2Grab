# Guide de Contribution

Document interne — Pour l'équipe

---

## Regles

1. **Ne pas pousser sur `main`** — Toujours utiliser une branche.
2. **Votre dossier personnel** — `sandbox/<votre_nom>/` pour vos tests.
3. **Pull Request** — Pour tout changement dans `src/`.

---

## Comment travailler avec Git

### 1. Cloner le projet (premiere fois seulement)

```bash
git clone https://github.com/dirobots/Vision2grab.git
cd Vision2grab
```

### 2. Creer une branche

```bash
# Aller sur main et mettre a jour
git checkout main
git pull origin main

# Creer votre branche
git checkout -b feature/<votre_nom>-<description>

# Exemple:
git checkout -b feature/nas-dagger
```

### 3. Faire vos modifications

Travaillez sur votre code, puis:

```bash
# Voir les fichiers modifies
git status

# Ajouter vos fichiers
git add .

# Ou ajouter un fichier specifique
git add chemin/vers/fichier.py

# Faire un commit
git commit -m "Description courte de vos changements"
```

### 4. Pousser votre branche

```bash
# Premiere fois sur cette branche
git push -u origin feature/<votre_nom>-<description>

# Les fois suivantes
git push
```

### 5. Creer une Pull Request

1. Aller sur GitHub
2. Cliquer sur "Compare & Pull Request"
3. Decrire vos changements
4. Demander une review

---

## Dossiers Personnels

| Membre | Dossier |
|--------|---------|
| Nas | `sandbox/nas/` |
| Pierre | `sandbox/pierre/` |
| Quan | `sandbox/quan/` |
| Josh | `sandbox/josh/` |
| Cassandre | `sandbox/cassandre/` |
| Oualid | `sandbox/oualid/` |
| Ariel | `sandbox/ariel/` |

---

## Avant une PR

- Code fonctionne localement
- Pas de fichiers inutiles
- Message de commit clair

---

## Conseil

Si vous n'etes pas certain de comment faire, utilisez un IDE avec IA comme **Cursor**, **Antigravity**, ou **GitHub Copilot**. Demandez-lui de suivre les instructions de `CONTRIBUTING.md` avant de push.
