# NLP- Homework 01 Corpus Analysis

## Project Overview

This project analyzes movie scripts from two distinct genres (Romantic and Sci-Fi) using Natural Language Processing techniques including Naive Bayes probability modeling and Latent Dirichlet Allocation (LDA) topic modeling.

## Dataset

### Source
Movie scripts collected from [The Internet Movie Script Database (IMSDb)](https://imsdb.com/)

### Categories and Films

**Romantic Category (5 films, 5,382 documents):**
- The Fault in Our Stars
- The Perks of Being a Wallflower
- Pride and Prejudice
- Titanic
- 10 Things I Hate About You

**Sci-Fi Category (5 films, 6,038 documents):**
- Avatar
- The Fifth Element
- Interstellar
- Tenet
- The Martian

**Total:** 10 films, 11,420 documents (script blocks)

### Document Structure
Each film script was split into smaller blocks (paragraphs/scenes) to create sufficient samples for analysis. Each block represents one document in the corpus.

## Methodology

### 1. Data Collection & Preprocessing
- Scripts downloaded from IMSDb using web scraping
- Each script split into paragraph-level blocks
- Organized into category-specific folders

### 2. Bag-of-Words Representation
- Text converted to lowercase
- English stop words removed
- Character names filtered (see experimentation section)
- Vocabulary size: 1,209 words (after aggressive filtering)
- Parameters: `min_df=10`, `max_df=0.7`, minimum word length=3

### 3. Naive Bayes Analysis
- Computed P(w|c) for each word in each category
- Applied add-one smoothing
- Calculated Log-Likelihood Ratio (LLR):
```
  llr(w, c) = log(P(w|c)) - log(P(w|Co))
```
- Used Fightin' Words method (Log Odds Ratio with Dirichlet Prior)
- Identified top 10 distinctive words per category

### 4. Topic Modeling (LDA)
- Latent Dirichlet Allocation with 10 topics
- Library: scikit-learn `LatentDirichletAllocation`
- Parameters: `n_components=10`, `random_state=42`, `max_iter=20`
- Manually labeled topics based on top words
- Computed average topic distribution per category

### 5. Experimentation
Tested three variations to optimize results:

**Experiment 1: Stemming**
- Applied Porter Stemmer
- Vocabulary reduced from 2,547 to 2,301 words
- Result: Minimal improvement in topic quality

**Experiment 2: TF-IDF**
- Used TF-IDF weighting instead of raw counts
- Same vocabulary size (2,547 words)
- Result: More balanced distributions but poor category separation

**Experiment 3: Aggressive Stopword Filtering** **BEST**
- Removed 100+ character names manually
- Stricter parameters: `min_df=10`, `max_df=0.7`, `min_length=3`
- Vocabulary reduced to 1,209 words
- Result: Cleanest topics, best interpretability, good category separation

## Key Results

### Top Distinctive Words

**Romantic Films:**
- bedroom, women, letter, aunt, afternoon, dress, chastity

**Sci-Fi Films:**
- probe, airlock, forest, banshee, hab, troopers, oxygen

### Topic Modeling (Aggressive Filtering - Best Configuration)

**Top 5 Topics for Romantic:**
1. Topic 4: Room & water scenes (11.49%)
2. Topic 7: Observation & movement (11.14%)
3. Topic 2: Head & deck scenes (10.35%)
4. Topic 9: Ship & eyes scenes (10.24%)
5. Topic 6: Face & moment scenes (10.18%)

**Top 5 Topics for Sci-Fi:**
1. Topic 9: Ship & spacecraft (11.67%)
2. Topic 7: Observation & technology (11.37%)
3. Topic 6: Action & looking (11.26%)
4. Topic 0: Physical actions (10.05%)
5. Topic 1: Combat & interaction (9.79%)

## Insights & Findings

### 1. Character Name Dominance
Without aggressive filtering, character names (rose, jack, cooper, neytiri) dominated all analyses, obscuring meaningful thematic patterns.

### 2. Preprocessing Impact
- **Stemming:** Reduced vocabulary by 10% but didn't significantly improve topic quality
- **TF-IDF:** Helped balance word importance but caused topics to overlap between categories
- **Aggressive Filtering:** Most effective approach, revealing thematic patterns rather than character-specific language

### 3. Genre Distinctions
- **Romantic films** emphasize: interpersonal spaces (bedroom, deck), emotional states, correspondence (letter)
- **Sci-Fi films** emphasize: technology (probe, airlock, hab), exploration (forest), military (troopers)

### 4. Topic Model Quality
LDA identified themes that crossed individual films (e.g., "ship scenes" appeared in both Titanic and sci-fi space films), demonstrating that topics capture semantic patterns rather than film-specific vocabulary.

### 5. Trade-offs
Aggressive filtering removed 52% of vocabulary but improved interpretability and reduced overfitting to specific films.

## Requirements
```
Python 3.8+
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
beautifulsoup4>=4.10.0
requests>=2.26.0
nltk>=3.6.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Collection
Open and run `get_data.ipynb`:
- Downloads 10 movie scripts from IMSDb using web scraping
- Splits each script into paragraph-level blocks
- Organizes blocks into category folders:
  - `corpus/romantic_blocks/` (5,382 documents)
  - `corpus/scifi_blocks/` (6,038 documents)

### Step 2: Analysis
Open and run `hw.ipynb`:

**Section 2.2 - Bag-of-Words:**
- Loads documents from corpus folders
- Creates sparse BOW matrix with CountVectorizer
- Saves processed data to `processed_data/`

**Section 2.3 - Naive Bayes:**
- Computes P(w|c) with add-one smoothing
- Calculates Log-Likelihood Ratios
- Applies Fightin' Words method
- Identifies top 10 distinctive words per category

**Section 2.4 - Topic Modeling:**
- Runs LDA with 10 topics
- Generates topic-word distributions
- Computes category-topic distributions
- Reports top 5 topics per category

**Section 2.5 - Experimentation:**
- Experiment 1: Stemming
- Experiment 2: TF-IDF
- Experiment 3: Aggressive stopword filtering
- Compares all approaches and selects best configuration

### Step 3: View Results
All outputs (tables, statistics, comparisons) are displayed directly in `hw.ipynb`

## Future Improvements

1. **Named Entity Recognition (NER):** Automatically detect and filter character names
2. **More Genres:** Expand to additional genres (horror, comedy, action)
3. **Temporal Analysis:** Analyze how language changes across film decades
4. **Bigrams/Trigrams:** Capture multi-word expressions (e.g., "outer space")
5. **Coherence Optimization:** Systematically optimize number of topics using coherence scores

## Author

Merve Filiz Baker

## License

This project is for educational purposes. Movie scripts are copyrighted by their respective owners.
