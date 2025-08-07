import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  PictureAsPdf as PdfIcon,
} from '@mui/icons-material';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../layouts/MainLayout';

const DashboardPage = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // Load files from localStorage on component mount
  React.useEffect(() => {
    const savedFiles = localStorage.getItem('pdfExtractorFiles');
    if (savedFiles) {
      try {
        const parsedFiles = JSON.parse(savedFiles);
        setFiles(parsedFiles);
      } catch (error) {
        console.error('Error loading files from localStorage:', error);
      }
    }
  }, []);

  // Save files to localStorage whenever files state changes
  React.useEffect(() => {
    localStorage.setItem('pdfExtractorFiles', JSON.stringify(files));
  }, [files]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || file.type !== 'application/pdf') {
      setSnackbar({ open: true, message: 'Vui lòng chọn file PDF!', severity: 'error' });
      return;
    }
    const tempId = Date.now();
    setFiles([{ id: tempId, name: file.name, size: file.size, status: 'processing', data: null }, ...files]);

    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:8080/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const extracted = res.data.data;
      setFiles(prev => prev.map(f => f.id === tempId ? { ...f, status: 'completed', data: extracted } : f));
      setSnackbar({ open: true, message: 'File đã được xử lý thành công!', severity: 'success' });
    } catch (err) {
      console.error(err);
      setFiles(prev => prev.map(f => f.id === tempId ? { ...f, status: 'error' } : f));
      setSnackbar({ open: true, message: 'Có lỗi xảy ra khi xử lý file!', severity: 'error' });
    }
  };

  const handleMenuOpen = (event, file) => {
    setSelectedFile(file);
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedFile(null);
  };

  const handleDelete = () => {
    if (selectedFile) {
      setFiles(files.filter(f => f.id !== selectedFile.id));
      handleMenuClose();
      setSnackbar({
        open: true,
        message: 'File đã được xóa!',
        severity: 'success',
      });
    }
  };

  const handleDownload = () => {
    if (!selectedFile || !selectedFile.data) { handleMenuClose(); return; }
    const blob = new Blob([JSON.stringify(selectedFile.data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedFile.name.replace(/\.pdf$/i, '')}_extracted.json`;
    a.click();
    URL.revokeObjectURL(url);
    handleMenuClose();
  };

  const handlePreview = () => {
    if (!selectedFile || !selectedFile.data) { handleMenuClose(); return; }
    navigate('/preview', { 
      state: { 
        data: selectedFile.data, 
        fileName: selectedFile.name 
      } 
    });
    handleMenuClose();
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <MainLayout>
      <Box>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard
        </Typography>

        <Paper
          sx={{
            p: 3,
            mb: 3,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            backgroundColor: 'primary.main',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            '&:hover': {
              backgroundColor: 'primary.dark',
              transform: 'translateY(-2px)',
            },
          }}
          component="label"
        >
          <input
            type="file"
            accept=".pdf"
            hidden
            onChange={handleFileUpload}
            disabled={loading}
          />
          <UploadIcon sx={{ fontSize: 48, color: 'white', mb: 1 }} />
          <Typography variant="h6" gutterBottom color="white">
            Tải lên file PDF
          </Typography>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
            Kéo thả file vào đây hoặc click để chọn file
          </Typography>
        </Paper>

        <Grid container spacing={3}>
          {files.map((file) => (
            <Grid item xs={12} sm={6} md={4} key={file.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  position: 'relative',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: 3,
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <PdfIcon sx={{ color: 'primary.main', mr: 1 }} />
                    <Typography
                      variant="subtitle1"
                      noWrap
                      sx={{ flexGrow: 1, fontWeight: 500 }}
                    >
                      {file.name}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, file)}
                    >
                      <MoreVertIcon />
                    </IconButton>
                  </Box>

                  <Typography
                    variant="body2"
                    color="text.secondary"
                    gutterBottom
                  >
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </Typography>

                  {file.status === 'processing' ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={16} />
                      <Typography variant="body2" color="text.secondary">
                        Đang xử lý...
                      </Typography>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="success.main">
                      Đã hoàn thành
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          PaperProps={{
            sx: { minWidth: 180 },
          }}
        >
          <MenuItem onClick={handlePreview}>
            <ListItemIcon>
              <ViewIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Xem trước" />
          </MenuItem>
          <MenuItem onClick={handleDownload}>
            <ListItemIcon>
              <DownloadIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Tải xuống" />
          </MenuItem>
          <MenuItem onClick={handleDelete}>
            <ListItemIcon>
              <DeleteIcon fontSize="small" color="error" />
            </ListItemIcon>
            <ListItemText primary="Xóa" sx={{ color: 'error.main' }} />
          </MenuItem>
        </Menu>

        <Snackbar
          open={snackbar.open}
          autoHideDuration={4000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={handleCloseSnackbar}
            severity={snackbar.severity}
            variant="filled"
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </MainLayout>
  );
};

export default DashboardPage; 