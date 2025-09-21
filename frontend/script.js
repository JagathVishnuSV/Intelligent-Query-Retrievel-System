document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('file-upload');
    const fileName = document.getElementById('file-name');
    const urlInput = document.getElementById('url-input');
    const questionsInput = document.getElementById('questions-input');
    const submitBtn = document.getElementById('submit-btn');
    const resultsContainer = document.getElementById('results-container');

    fileUpload.addEventListener('change', () => {
        if (fileUpload.files.length > 0) {
            fileName.textContent = fileUpload.files[0].name;
            urlInput.value = ''; // Clear URL input if a file is chosen
        } else {
            fileName.textContent = 'No file chosen';
        }
    });

    urlInput.addEventListener('input', () => {
        if (urlInput.value) {
            fileUpload.value = ''; // Clear file input if a URL is entered
            fileName.textContent = 'No file chosen';
        }
    });

    submitBtn.addEventListener('click', async () => {
        const file = fileUpload.files[0];
        const url = urlInput.value.trim();
        const questions = questionsInput.value.trim();

        if ((!file && !url) || !questions) {
            alert('Please provide a document (file or URL) and at least one question.');
            return;
        }

        const formData = new FormData();
        formData.append('questions', questions);

        if (file) {
            formData.append('document_file', file);
        } else if (url) {
            formData.append('document_url', url);
        }

        resultsContainer.innerHTML = '<p>Loading...</p>';

        try {
            const response = await fetch('/api/v1/hackrx/run', {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer 714c3fdb7fd84d510e3b5d4a0e21cc85a9a323700c63fad79fcd234ea93b99d5'
                },
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred');
            }

            const result = await response.json();
            displayResults(result.answers);

        } catch (error) {
            resultsContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    });

    function displayResults(answers) {
        resultsContainer.innerHTML = '';
        if (answers && answers.length > 0) {
            const ul = document.createElement('ul');
            answers.forEach(answer => {
                const li = document.createElement('li');
                li.textContent = answer;
                ul.appendChild(li);
            });
            resultsContainer.appendChild(ul);
        } else {
            resultsContainer.innerHTML = '<p class="no-results">No answers were returned.</p>';
        }
    }
});
