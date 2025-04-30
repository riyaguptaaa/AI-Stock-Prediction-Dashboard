document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Make 3D plot responsive
    function resizePlots() {
        if (typeof Plotly !== 'undefined') {
            const plotContainers = document.querySelectorAll('.chart-container');
            plotContainers.forEach(container => {
                if (container.querySelector('.plotly-graph-div')) {
                    Plotly.Plots.resize(container.querySelector('.plotly-graph-div'));
                }
            });
        }
    }

    // Initial resize
    resizePlots();
    
    // Resize on window change
    window.addEventListener('resize', resizePlots);
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Animate metric cards on scroll
    const observerOptions = {
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.metric-card').forEach(card => {
        observer.observe(card);
    });
});