// Общая логика для сайдбара

// Прокрутка к секции
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
        updateActiveNavLink(sectionId);
    }
}

// Обновление активной ссылки в навигации
function updateActiveNavLink(activeId) {
    // Убираем активный класс со всех ссылок
    document.querySelectorAll('.sidebar-nav a').forEach(link => {
        link.classList.remove('active');
    });
    
    // Добавляем активный класс к текущей ссылке
    const activeLink = document.querySelector(`.sidebar-nav a[href="#${activeId}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
}

// Переключение мобильного меню
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('open');
}

// Обновление активной ссылки при прокрутке
function handleScroll() {
    // Получаем все секции из навигации
    const navLinks = document.querySelectorAll('.sidebar-nav a');
    const sections = [];
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && href.startsWith('#')) {
            const sectionId = href.substring(1);
            const element = document.getElementById(sectionId);
            if (element) {
                sections.push(sectionId);
            }
        }
    });
    
    const scrollPos = window.scrollY + 100; // offset для точности
    
    for (let sectionId of sections) {
        const element = document.getElementById(sectionId);
        if (element) {
            const top = element.offsetTop;
            const bottom = top + element.offsetHeight;
            
            if (scrollPos >= top && scrollPos < bottom) {
                updateActiveNavLink(sectionId);
                break;
            }
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Добавляем обработчик прокрутки
    window.addEventListener('scroll', handleScroll);
    
    // Инициализация активной ссылки при загрузке
    setTimeout(() => {
        handleScroll();
    }, 100);
    
    // Закрытие мобильного меню при клике вне его
    document.addEventListener('click', function(event) {
        const sidebar = document.querySelector('.sidebar');
        const menuBtn = document.querySelector('.mobile-menu-btn');
        
        if (sidebar && menuBtn && sidebar.classList.contains('open')) {
            if (!sidebar.contains(event.target) && !menuBtn.contains(event.target)) {
                sidebar.classList.remove('open');
            }
        }
    });
});
