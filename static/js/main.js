const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const predictBtn = document.getElementById('predictBtn');
const fileName = document.getElementById('fileName');
const imagePreview = document.getElementById('imagePreview');
const preview = document.getElementById('preview');
const loader = document.getElementById('loader');
const result = document.getElementById('result');

let selectedFile = null;

uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        fileName.textContent = `Selected: ${selectedFile.name}`;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            predictBtn.classList.remove('hidden');
        };
        reader.readAsDataURL(selectedFile);

        // Hide previous results
        result.classList.add('hidden');
    }
});

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first!');
        return;
    }

    // Show loader
    loader.classList.remove('hidden');
    predictBtn.disabled = true;
    result.classList.add('hidden');

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }
});

function displayResult(data) {
    document.getElementById('prediction').textContent =
        `Prediction: ${data.prediction}`;

    document.getElementById('confidence').textContent =
        `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    const probabilities = document.getElementById('probabilities');

    probabilities.innerHTML = '';

    for (const [label, prob] of Object.entries(data.probabilities)) {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            ${label}:
            ${(prob * 100).toFixed(2)}%
        `;
        probabilities.appendChild(probItem);
    }

    result.classList.remove('hidden');
}