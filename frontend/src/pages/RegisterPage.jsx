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
import AuthLayout from '../layouts/AuthLayout';

const validationSchema = yup.object({
  fullName: yup.string().required('Họ tên là bắt buộc'),
  email: yup
    .string()
    .email('Email không hợp lệ')
    .required('Email là bắt buộc'),
  password: yup
    .string()
    .min(6, 'Mật khẩu phải có ít nhất 6 ký tự')
    .required('Mật khẩu là bắt buộc'),
  confirmPassword: yup
    .string()
    .oneOf([yup.ref('password'), null], 'Mật khẩu không khớp')
    .required('Xác nhận mật khẩu là bắt buộc'),
});

const RegisterPage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = React.useState(false);

  const formik = useFormik({
    initialValues: {
      fullName: '',
      email: '',
      password: '',
      confirmPassword: '',
    },
    validationSchema: validationSchema,
    onSubmit: async (values) => {
      setLoading(true);
      try {
        // TODO: Implement registration logic
        console.log('Register values:', values);
        navigate('/dashboard');
      } catch (error) {
        console.error('Registration error:', error);
      } finally {
        setLoading(false);
      }
    },
  });

  const googleLogin = useGoogleLogin({
    onSuccess: async (response) => {
      try {
        // TODO: Send token to backend
        console.log('Google login success:', response);
        navigate('/dashboard');
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
          Đăng ký tài khoản
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Tạo tài khoản mới để sử dụng dịch vụ
        </Typography>
      </Box>

      <form onSubmit={formik.handleSubmit}>
        <TextField
          fullWidth
          id="fullName"
          name="fullName"
          label="Họ và tên"
          variant="outlined"
          margin="normal"
          value={formik.values.fullName}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.fullName && Boolean(formik.errors.fullName)}
          helperText={formik.touched.fullName && formik.errors.fullName}
          disabled={loading}
        />

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

        <TextField
          fullWidth
          id="confirmPassword"
          name="confirmPassword"
          label="Xác nhận mật khẩu"
          type="password"
          variant="outlined"
          margin="normal"
          value={formik.values.confirmPassword}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={
            formik.touched.confirmPassword &&
            Boolean(formik.errors.confirmPassword)
          }
          helperText={
            formik.touched.confirmPassword && formik.errors.confirmPassword
          }
          disabled={loading}
        />

        <Button
          fullWidth
          type="submit"
          variant="contained"
          size="large"
          sx={{ mt: 3 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Đăng ký'}
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
          Đăng ký bằng Google
        </Button>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="body2">
            Đã có tài khoản?{' '}
            <Link
              to="/login"
              style={{ textDecoration: 'none', color: 'primary.main' }}
            >
              <Typography
                component="span"
                variant="body2"
                color="primary"
                sx={{ '&:hover': { textDecoration: 'underline' } }}
              >
                Đăng nhập
              </Typography>
            </Link>
          </Typography>
        </Box>
      </form>
    </AuthLayout>
  );
};

export default RegisterPage; 