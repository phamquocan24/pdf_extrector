import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  CircularProgress,
  Alert,
  Paper,
  InputAdornment,
  IconButton,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { useFormik } from 'formik';
import * as yup from 'yup';
import axios from 'axios';
import AuthLayout from '../layouts/AuthLayout';

const validationSchema = yup.object({
  password: yup
    .string()
    .min(6, 'Mật khẩu phải có ít nhất 6 ký tự')
    .required('Mật khẩu là bắt buộc'),
  confirmPassword: yup
    .string()
    .oneOf([yup.ref('password'), null], 'Mật khẩu xác nhận không khớp')
    .required('Xác nhận mật khẩu là bắt buộc'),
});

const ResetPasswordPage = () => {
  const { token } = useParams();
  const navigate = useNavigate();
  
  const [loading, setLoading] = useState(false);
  const [verifying, setVerifying] = useState(true);
  const [tokenValid, setTokenValid] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Verify token on component mount
  useEffect(() => {
    const verifyToken = async () => {
      try {
        const response = await axios.get(`http://localhost:8080/api/auth/verify-reset-token/${token}`);
        if (response.data.valid) {
          setTokenValid(true);
          setUserEmail(response.data.email);
        } else {
          setError('Token đặt lại mật khẩu không hợp lệ hoặc đã hết hạn');
        }
      } catch (error) {
        console.error('Token verification error:', error);
        setError('Token đặt lại mật khẩu không hợp lệ hoặc đã hết hạn');
      } finally {
        setVerifying(false);
      }
    };

    if (token) {
      verifyToken();
    } else {
      setError('Không tìm thấy token đặt lại mật khẩu');
      setVerifying(false);
    }
  }, [token]);

  const formik = useFormik({
    initialValues: {
      password: '',
      confirmPassword: '',
    },
    validationSchema: validationSchema,
    onSubmit: async (values) => {
      setLoading(true);
      setError('');
      try {
        const response = await axios.post(`http://localhost:8080/api/auth/reset-password/${token}`, {
          password: values.password
        });
        
        if (response.data.success) {
          setSuccess(true);
          // Redirect to login after 3 seconds
          setTimeout(() => {
            navigate('/login');
          }, 3000);
        }
      } catch (error) {
        console.error('Reset password error:', error);
        const errorMessage = error.response?.data?.message || 'Có lỗi xảy ra. Vui lòng thử lại.';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    },
  });

  const handleClickShowPassword = () => {
    setShowPassword(!showPassword);
  };

  const handleClickShowConfirmPassword = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  // Loading state while verifying token
  if (verifying) {
    return (
      <AuthLayout>
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CircularProgress size={40} />
          <Typography variant="body1" sx={{ mt: 2 }}>
            Đang xác thực token...
          </Typography>
        </Box>
      </AuthLayout>
    );
  }

  // Success state
  if (success) {
    return (
      <AuthLayout>
        <Box sx={{ textAlign: 'center' }}>
          <CheckCircleIcon sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
          <Typography variant="h5" component="h1" color="success.main" gutterBottom>
            Đặt lại mật khẩu thành công!
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Mật khẩu của bạn đã được cập nhật thành công.
            Bạn sẽ được chuyển đến trang đăng nhập trong giây lát...
          </Typography>
          <Button
            component={Link}
            to="/login"
            variant="contained"
            sx={{ mt: 2 }}
          >
            Đăng nhập ngay
          </Button>
        </Box>
      </AuthLayout>
    );
  }

  // Invalid token state
  if (!tokenValid) {
    return (
      <AuthLayout>
        <Box sx={{ textAlign: 'center' }}>
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
          <Typography variant="h6" gutterBottom>
            Không thể đặt lại mật khẩu
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Token có thể đã hết hạn hoặc không hợp lệ. Vui lòng yêu cầu đặt lại mật khẩu mới.
          </Typography>
          <Button
            component={Link}
            to="/forgot-password"
            variant="contained"
            sx={{ mr: 2 }}
          >
            Yêu cầu mới
          </Button>
          <Button
            component={Link}
            to="/login"
            variant="outlined"
          >
            Quay lại đăng nhập
          </Button>
        </Box>
      </AuthLayout>
    );
  }

  // Reset password form
  return (
    <AuthLayout>
      <Box sx={{ textAlign: 'center', mb: 3 }}>
        <Typography variant="h5" component="h1" color="primary" gutterBottom>
          Đặt lại mật khẩu
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Nhập mật khẩu mới cho tài khoản: <strong>{userEmail}</strong>
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
          id="password"
          name="password"
          label="Mật khẩu mới"
          type={showPassword ? 'text' : 'password'}
          variant="outlined"
          margin="normal"
          value={formik.values.password}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.password && Boolean(formik.errors.password)}
          helperText={formik.touched.password && formik.errors.password}
          disabled={loading}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label="toggle password visibility"
                  onClick={handleClickShowPassword}
                  edge="end"
                >
                  {showPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />

        <TextField
          fullWidth
          id="confirmPassword"
          name="confirmPassword"
          label="Xác nhận mật khẩu mới"
          type={showConfirmPassword ? 'text' : 'password'}
          variant="outlined"
          margin="normal"
          value={formik.values.confirmPassword}
          onChange={formik.handleChange}
          onBlur={formik.handleBlur}
          error={formik.touched.confirmPassword && Boolean(formik.errors.confirmPassword)}
          helperText={formik.touched.confirmPassword && formik.errors.confirmPassword}
          disabled={loading}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label="toggle confirm password visibility"
                  onClick={handleClickShowConfirmPassword}
                  edge="end"
                >
                  {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />

        <Button
          fullWidth
          type="submit"
          variant="contained"
          size="large"
          sx={{ mt: 3 }}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Đặt lại mật khẩu'}
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

export default ResetPasswordPage;