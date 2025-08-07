import React from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Divider,
  CircularProgress,
} from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import { useFormik } from 'formik';
import * as yup from 'yup';
import { useGoogleLogin } from '@react-oauth/google';
import { useAuth } from '../contexts/AuthContext'; // Import useAuth
import AuthLayout from '../layouts/AuthLayout';

const validationSchema = yup.object({
  email: yup
    .string()
    .email('Email không hợp lệ')
    .required('Email là bắt buộc'),
  password: yup
    .string()
    .min(6, 'Mật khẩu phải có ít nhất 6 ký tự')
    .required('Mật khẩu là bắt buộc'),
});

const LoginPage = () => {
  const { login } = useAuth(); // Get login function from context
  const navigate = useNavigate();
  const [loading, setLoading] = React.useState(false);

  const formik = useFormik({
    initialValues: {
      email: '',
      password: '',
    },
    validationSchema: validationSchema,
    onSubmit: async (values) => {
      setLoading(true);
      try {
        // This is a mock login. Replace with your actual API call.
        console.log('Login values:', values);
        const mockUser = { name: values.email.split('@')[0], email: values.email, picture: '' };
        const mockToken = 'mock-jwt-token-for-email-login';
        login(mockUser, mockToken); // Use the login function from context
      } catch (error) {
        console.error('Login error:', error);
      } finally {
        setLoading(false);
      }
    },
  });

  const googleLogin = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      try {
        // Fetch user info from Google
        const userInfoResponse = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
          headers: { Authorization: `Bearer ${tokenResponse.access_token}` },
        });
        const userInfo = await userInfoResponse.json();
        
        // In a real app, you would send this info to your backend to get a JWT token.
        // For now, we'll use the info directly.
        const appToken = tokenResponse.access_token; // Or a token from your backend
        login(userInfo, appToken); // Use the login function from context

      } catch (error) {
        console.error('Google login error:', error);
      }
    },
    onError: () => {
      console.error('Google login failed');
    },
  });

  return (
    <AuthLayout>
      <Box sx={{ textAlign: 'center', mb: 3 }}>
        <Typography variant="h5" component="h1" color="primary" gutterBottom>
          Đăng nhập
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Chào mừng bạn quay trở lại!
        </Typography>
      </Box>

      <form onSubmit={formik.handleSubmit}>
        <TextField
          fullWidth
          id="email"
          name="email"
          label="Email"
          variant="outlined"
          margin="normal"
          value={formik.values.email}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.email && Boolean(formik.errors.email)}
          helperText={formik.touched.email && formik.errors.email}
          disabled={loading}
        />

        <TextField
          fullWidth
          id="password"
          name="password"
          label="Mật khẩu"
          type="password"
          variant="outlined"
          margin="normal"
          value={formik.values.password}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.password && Boolean(formik.errors.password)}
          helperText={formik.touched.password && formik.errors.password}
          disabled={loading}
        />

        <Box sx={{ textAlign: 'right', mt: 1 }}>
          <Link
            to="/forgot-password"
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            <Typography
              variant="body2"
              color="primary"
              sx={{ '&:hover': { textDecoration: 'underline' } }}
            >
              Quên mật khẩu?
            </Typography>
          </Link>
        </Box>

        <Button
          fullWidth
          type="submit"
          variant="contained"
          size="large"
          sx={{ mt: 3 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Đăng nhập'}
        </Button>

        <Divider sx={{ my: 3 }}>
          <Typography variant="body2" color="text.secondary">
            HOẶC
          </Typography>
        </Divider>

        <Button
          fullWidth
          variant="outlined"
          size="large"
          onClick={() => googleLogin()}
          disabled={loading}
          sx={{
            color: 'text.primary',
            borderColor: 'divider',
            '&:hover': {
              borderColor: 'primary.main',
            },
          }}
        >
          <img
            src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
            alt="Google"
            style={{ width: 20, height: 20, marginRight: 8 }}
          />
          Đăng nhập bằng Google
        </Button>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="body2">
            Chưa có tài khoản?{' '}
            <Link
              to="/register"
              style={{ textDecoration: 'none', color: 'primary.main' }}
            >
              <Typography
                component="span"
                variant="body2"
                color="primary"
                sx={{ '&:hover': { textDecoration: 'underline' } }}
              >
                Đăng ký ngay
              </Typography>
            </Link>
          </Typography>
        </Box>
      </form>
    </AuthLayout>
  );
};

export default LoginPage; 