import React from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Link } from 'react-router-dom';
import { useFormik } from 'formik';
import * as yup from 'yup';
import axios from 'axios';
import AuthLayout from '../layouts/AuthLayout';

const validationSchema = yup.object({
  email: yup
    .string()
    .email('Email không hợp lệ')
    .required('Email là bắt buộc'),
});

const ForgotPasswordPage = () => {
  const [loading, setLoading] = React.useState(false);
  const [sent, setSent] = React.useState(false);
  const [error, setError] = React.useState('');

  const formik = useFormik({
    initialValues: {
      email: '',
    },
    validationSchema: validationSchema,
    onSubmit: async (values) => {
      setLoading(true);
      setError('');
      try {
        const response = await axios.post('http://localhost:8080/api/auth/forgot-password', {
          email: values.email
        });
        
        if (response.data.success) {
          setSent(true);
        }
      } catch (error) {
        console.error('Forgot password error:', error);
        const errorMessage = error.response?.data?.message || 'Có lỗi xảy ra. Vui lòng thử lại.';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    },
  });

  if (sent) {
    return (
      <AuthLayout>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h5" component="h1" color="primary" gutterBottom>
            Email đã được gửi!
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Chúng tôi đã gửi hướng dẫn khôi phục mật khẩu đến email của bạn.
            Vui lòng kiểm tra hộp thư đến.
          </Typography>
          <Button
            component={Link}
            to="/login"
            variant="contained"
            sx={{ mt: 2 }}
          >
            Quay lại đăng nhập
          </Button>
        </Box>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout>
      <Box sx={{ textAlign: 'center', mb: 3 }}>
        <Typography variant="h5" component="h1" color="primary" gutterBottom>
          Quên mật khẩu
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Nhập email của bạn để khôi phục mật khẩu
        </Typography>
      </Box>

      <form onSubmit={formik.handleSubmit}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
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

        <Button
          fullWidth
          type="submit"
          variant="contained"
          size="large"
          sx={{ mt: 3 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Gửi yêu cầu'}
        </Button>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="body2">
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
                Quay lại đăng nhập
              </Typography>
            </Link>
          </Typography>
        </Box>
      </form>
    </AuthLayout>
  );
};

export default ForgotPasswordPage; 