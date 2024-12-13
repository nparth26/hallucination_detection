async function analyzeQuery() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) {
        alert('Please enter a query');
        return;
    }

    showLoading(true);
    try {
        const response = await fetch('/analyze_complete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your query');
    } finally {
        showLoading(false);
    }
}

function checkHallucination(scores) {
    for (const score of scores) {
        if (score.similarity_score < 5) {
            return {
                hasHallucination: true,
                concept: score.original_concept,
                score: score.similarity_score
            };
        }
    }
    return { hasHallucination: false };
}

function displayResults(data) {
    // Display concepts
    const conceptsDisplay = document.getElementById('conceptsDisplay');
    conceptsDisplay.innerHTML = data.concepts.map(concept => 
        `<div class="concept-item">${concept}</div>`
    ).join('');

    // Display definitions
    const definitionsDisplay = document.getElementById('definitionsDisplay');
    definitionsDisplay.innerHTML = data.definitions.map(def => 
        `<div class="definition-item">
            <strong>${def.concept}:</strong>
            <p>${def.masked_definition}</p>
        </div>`
    ).join('');

    // Display inferences
    const inferencesDisplay = document.getElementById('inferencesDisplay');
    inferencesDisplay.innerHTML = data.inferences.map(inf => 
        `<div class="inference-item">
            <p><strong>Original:</strong> ${inf.original_concept}</p>
            <p><strong>Inferred:</strong> ${inf.inferred_concept}</p>
        </div>`
    ).join('');

    // Display similarity scores
    const scoresDisplay = document.getElementById('scoresDisplay');
    scoresDisplay.innerHTML = data.similarity_scores.map(score => 
        `<div class="score-item">
            <p><strong>${score.original_concept}</strong></p>
            <p>Similarity Score: ${score.similarity_score}/10</p>
        </div>`
    ).join('');


    const finalResultDisplay = document.getElementById('finalResultDisplay');
    const hallucination = checkHallucination(data.similarity_scores);
    
    if (hallucination.hasHallucination) {
        finalResultDisplay.innerHTML = `The model will hallucinate as the familiarity score for the concept "${hallucination.concept}" is very less (${hallucination.score.toFixed(2)}/10).`;
        finalResultDisplay.className = 'warning-result';
    } else {
        finalResultDisplay.innerHTML = 'The model is familiar with all the concepts in the query and Hallucination is very less likely.';
        finalResultDisplay.className = 'success-result';
    }
}

function clearAll() {
    document.getElementById('queryInput').value = '';
    document.getElementById('conceptsDisplay').innerHTML = '';
    document.getElementById('definitionsDisplay').innerHTML = '';
    document.getElementById('inferencesDisplay').innerHTML = '';
    document.getElementById('scoresDisplay').innerHTML = '';
}

function showLoading(show) {
    const loader = document.getElementById('loadingIndicator');
    loader.style.display = show ? 'block' : 'none';
}