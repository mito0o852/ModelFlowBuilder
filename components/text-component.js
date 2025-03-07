// Fix text component rendering
function renderTextComponent(element, text, description) {
  // ... existing code ...
  
  // Ensure text doesn't overflow
  const textElement = document.createElement('div');
  textElement.className = 'component-title';
  textElement.textContent = text;
  
  const descriptionElement = document.createElement('div');
  descriptionElement.className = 'component-description';
  descriptionElement.textContent = description;
  
  // Apply proper text truncation with ellipsis
  element.style.overflow = 'hidden';
  textElement.style.textOverflow = 'ellipsis';
  textElement.style.whiteSpace = 'nowrap';
  
  // For descriptions, allow wrapping to two lines with ellipsis
  descriptionElement.style.display = '-webkit-box';
  descriptionElement.style.webkitLineClamp = '2';
  descriptionElement.style.webkitBoxOrient = 'vertical';
  descriptionElement.style.overflow = 'hidden';
  descriptionElement.style.textOverflow = 'ellipsis';
  
  element.appendChild(textElement);
  element.appendChild(descriptionElement);
  
  // ... existing code ...
} 