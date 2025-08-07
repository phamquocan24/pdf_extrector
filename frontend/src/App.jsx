import React from 'react';
import { Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { Box, CssBaseline } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { AuthProvider, useAuth } from './contexts/AuthContext'; // Import useAuth

// Import theme
import theme from './theme';

// Import pages
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import DashboardPage from './pages/DashboardPage';
import SettingsPage from './pages/SettingsPage';
import ProfilePage from './pages/ProfilePage';
import PreviewPage from './pages/PreviewPage';

// Get Google Client ID from environment variables
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

if (!GOOGLE_CLIENT_ID) {
  console.error("Missing Google Client ID. Please set VITE_GOOGLE_CLIENT_ID in your .env file.");
} else {
  console.log("Using Google Client ID:", GOOGLE_CLIENT_ID);
}
// A simple layout for protected routes
const ProtectedLayout = () => {
  const { user, token } = useAuth();

  if (!user && !token) {
    // If no user and no token, redirect to login
    return <Navigate to="/login" replace />;
  }

  // If we have a user, render the child routes
  return (
    <Box>
      {/* You can add a shared navbar/sidebar here */}
      <Outlet />
    </Box>
  );
};

function App() {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <AuthProvider>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Box sx={{ minHeight: '100vh' }}>
            <Routes>
              {/* Auth routes */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/forgot-password" element={<ForgotPasswordPage />} />
              <Route path="/reset-password/:token" element={<ResetPasswordPage />} />
              
              {/* Protected routes */}
              <Route path="/" element={<ProtectedLayout />}>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="preview" element={<PreviewPage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="profile" element={<ProfilePage />} />
              </Route>

              {/* Redirect unknown routes to login for now */}
              <Route path="*" element={<Navigate to="/login" replace />} />
            </Routes>
          </Box>
        </ThemeProvider>
      </AuthProvider>
    </GoogleOAuthProvider>
  );
}

export default App; 