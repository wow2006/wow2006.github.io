---
title: "NVIDIA GPU Architectures Reference"
date: 2025-11-02
draft: false
description: "Complete reference for NVIDIA GPU architectures with search functionality"
---

<style>
.search-container {
    margin: 20px 0;
}
.search-input {
    width: 100%;
    padding: 10px;
    font-size: 16px;
    border: 2px solid #76b900;
    border-radius: 4px;
}
.arch-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
    font-size: 14px;
}
.arch-table th {
    background-color: #76b900;
    color: white;
    padding: 12px 8px;
    text-align: left;
    font-weight: 600;
}
.arch-table td {
    padding: 10px 8px;
    border-bottom: 1px solid #ddd;
    vertical-align: top;
}
.arch-table tr:hover {
    background-color: #f5f5f5;
}
.arch-table a {
    color: #76b900;
    text-decoration: none;
}
.arch-table a:hover {
    text-decoration: underline;
}
.no-results {
    text-align: center;
    padding: 20px;
    color: #666;
    display: none;
}
.gpu-list {
    line-height: 1.6;
}
.loading {
    text-align: center;
    padding: 20px;
    color: #666;
}
</style>

<div class="search-container">
    <input type="text" id="searchInput" class="search-input" placeholder="Search by architecture, GPU series, SM version, compute capability, or code name...">
</div>

<div class="loading" id="loading">Loading GPU architectures...</div>

<table class="arch-table" id="archTable" style="display: none;">
    <thead>
        <tr>
            <th>Architecture</th>
            <th>Release Year</th>
            <th>SM Version</th>
            <th>Compute Capability</th>
            <th>GPU Code Names</th>
            <th>Example GPU Series</th>
            <th>Whitepaper</th>
        </tr>
    </thead>
    <tbody id="tableBody">
    </tbody>
</table>

<div class="no-results" id="noResults">No matching architectures found.</div>

<script>
// Path to your JSON file - adjust this to your Hugo static folder path
const JSON_DATA_PATH = '/data/nvidia-architectures.json';

async function loadArchitectures() {
    try {
        const response = await fetch(JSON_DATA_PATH);
        const data = await response.json();
        
        const tbody = document.getElementById('tableBody');
        const loading = document.getElementById('loading');
        const table = document.getElementById('archTable');
        
        // Clear loading message
        loading.style.display = 'none';
        table.style.display = 'table';
        
        // Populate table
        data.architectures.forEach(arch => {
            const row = document.createElement('tr');
            
            // Create GPU series list
            const gpuList = arch.gpuSeries.map(gpu => gpu).join('<br>');
            
            row.innerHTML = `
                <td>${arch.name}</td>
                <td>${arch.releaseYear}</td>
                <td>${arch.smVersion}</td>
                <td>${arch.computeCapability}</td>
                <td>${arch.codeNames.join(', ')}</td>
                <td class="gpu-list">${gpuList}</td>
                <td><a href="${arch.whitepaper}" target="_blank">${arch.name} Whitepaper</a></td>
            `;
            
            tbody.appendChild(row);
        });
        
        // Enable search after data is loaded
        setupSearch();
        
    } catch (error) {
        console.error('Error loading architecture data:', error);
        document.getElementById('loading').textContent = 'Error loading data. Please check the JSON file path.';
    }
}

function setupSearch() {
    const searchInput = document.getElementById('searchInput');
    const table = document.getElementById('archTable');
    const tbody = document.getElementById('tableBody');
    const rows = tbody.getElementsByTagName('tr');
    const noResults = document.getElementById('noResults');

    searchInput.addEventListener('keyup', function() {
        const filter = searchInput.value.toLowerCase();
        let visibleCount = 0;

        for (let i = 0; i < rows.length; i++) {
            const row = rows[i];
            const text = row.textContent.toLowerCase();
            
            if (text.includes(filter)) {
                row.style.display = '';
                visibleCount++;
            } else {
                row.style.display = 'none';
            }
        }

        if (visibleCount === 0) {
            noResults.style.display = 'block';
            table.style.display = 'none';
        } else {
            noResults.style.display = 'none';
            table.style.display = 'table';
        }
    });
}

// Load data when page loads
document.addEventListener('DOMContentLoaded', loadArchitectures);
</script>

## Additional Resources

- [CUDA Compute Capability Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [NVIDIA Architecture Overview](https://developer.nvidia.com/cuda-gpus)

## Notes

**Compute Capability** refers to the version of the CUDA architecture supported by the GPU. Higher compute capabilities generally support more advanced features and better performance.

**SM Version** (Streaming Multiprocessor) indicates the internal architecture generation of the GPU cores.

**GPU Code Names** are the internal chip designations used by NVIDIA during development and manufacturing.

**GPU Series** shows representative consumer (GeForce/RTX), professional (Quadro/RTX), and datacenter (Tesla/A-series/H-series) GPUs for each architecture.

---

**Data Source**: The architecture data is loaded from `/data/nvidia-architectures.json`. Edit that file to update the information displayed on this page.