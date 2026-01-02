# AdeptAI Frontend

Modern React-based frontend for the AdeptAI recruitment system.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+
- npm or yarn

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm start
```

### Building for Production
```bash
# Build the application
npm run build

# Serve the built application
npm run serve
```

## 🏗️ Architecture

### Components
- **Dashboard**: Main overview and analytics
- **Search**: Candidate search interface
- **Candidates**: Candidate management
- **Analytics**: Performance metrics and insights
- **Settings**: Configuration and preferences

### Key Features
- **Responsive Design**: Mobile-first approach
- **Real-time Updates**: Live data synchronization
- **Advanced Search**: Semantic search with filters
- **Analytics Dashboard**: Performance metrics and insights
- **Dark Mode**: Theme switching capability

## 🔧 Development

### Project Structure
```
src/
├── components/          # Reusable UI components
├── pages/              # Page components
├── services/           # API services
├── context/            # React context
├── styles/             # CSS and styling
└── utils/              # Utility functions
```

### Available Scripts
- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run lint` - Run linting
- `npm run format` - Format code

## 🎨 Styling

### Design System
- **Colors**: Consistent color palette
- **Typography**: Modern font stack
- **Spacing**: 8px grid system
- **Components**: Reusable UI components

### Themes
- **Light Mode**: Default theme
- **Dark Mode**: Dark theme option
- **High Contrast**: Accessibility theme

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### Features
- **Mobile Navigation**: Collapsible menu
- **Touch Gestures**: Swipe and tap support
- **Responsive Images**: Optimized for all devices

## 🔌 API Integration

### Endpoints
- **Search**: `/api/search` - Candidate search
- **Health**: `/api/health` - System health
- **Performance**: `/api/performance` - Metrics
- **Candidates**: `/api/candidates` - Candidate data

### Error Handling
- **Network Errors**: Retry mechanisms
- **Validation Errors**: User-friendly messages
- **Loading States**: Progress indicators

## 🧪 Testing

### Test Types
- **Unit Tests**: Component testing
- **Integration Tests**: API integration
- **E2E Tests**: Full user workflows
- **Visual Tests**: UI regression testing

### Running Tests
```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

## 🚀 Deployment

### Build Optimization
- **Code Splitting**: Lazy loading
- **Tree Shaking**: Remove unused code
- **Minification**: Compress assets
- **Caching**: Browser caching strategies

### Production Build
```bash
# Build for production
npm run build

# Analyze bundle size
npm run analyze

# Deploy to server
npm run deploy
```

## 📊 Performance

### Optimization
- **Lazy Loading**: Component-based splitting
- **Memoization**: Prevent unnecessary re-renders
- **Virtual Scrolling**: Large list performance
- **Image Optimization**: WebP and lazy loading

### Monitoring
- **Core Web Vitals**: Performance metrics
- **Bundle Analysis**: Size optimization
- **Error Tracking**: Production monitoring

## 🔒 Security

### Security Features
- **XSS Protection**: Input sanitization
- **CSRF Protection**: Token validation
- **Content Security Policy**: Resource restrictions
- **Secure Headers**: HTTP security headers

### Best Practices
- **Input Validation**: Client-side validation
- **Secure Storage**: Local storage security
- **API Security**: Secure API communication
- **Authentication**: Secure user sessions

---

**AdeptAI Frontend** - Modern, responsive, and secure recruitment interface.