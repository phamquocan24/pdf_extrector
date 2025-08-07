import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  FormControlLabel,
  Radio,
  RadioGroup,
  Switch,
  Divider,
  Button,
  Alert,
  Snackbar,
} from '@mui/material';
import MainLayout from '../layouts/MainLayout';

const SettingsPage = () => {
  const [settings, setSettings] = useState({
    outputFormat: 'json',
    autoDownload: true,
    preserveFormatting: true,
    includeMetadata: true,
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success',
  });

  const handleFormatChange = (event) => {
    setSettings({
      ...settings,
      outputFormat: event.target.value,
    });
  };

  const handleSwitchChange = (name) => (event) => {
    setSettings({
      ...settings,
      [name]: event.target.checked,
    });
  };

  const handleSave = () => {
    // TODO: Implement settings save logic
    console.log('Saving settings:', settings);
    setSnackbar({
      open: true,
      message: 'Đã lưu cài đặt thành công!',
      severity: 'success',
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <MainLayout>
      <Box>
        <Typography variant="h4" component="h1" gutterBottom>
          Cài đặt
        </Typography>

        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Định dạng xuất file
          </Typography>
          <FormControl component="fieldset">
            <RadioGroup
              value={settings.outputFormat}
              onChange={handleFormatChange}
            >
              <FormControlLabel
                value="json"
                control={<Radio />}
                label={
                  <Box>
                    <Typography variant="body1">JSON</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Xuất dữ liệu dưới dạng cấu trúc JSON
                    </Typography>
                  </Box>
                }
              />
              <FormControlLabel
                value="csv"
                control={<Radio />}
                label={
                  <Box>
                    <Typography variant="body1">CSV</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Xuất dữ liệu dưới dạng bảng tính
                    </Typography>
                  </Box>
                }
              />
              <FormControlLabel
                value="doc"
                control={<Radio />}
                label={
                  <Box>
                    <Typography variant="body1">DOC</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Xuất dữ liệu dưới dạng tài liệu Word
                    </Typography>
                  </Box>
                }
              />
            </RadioGroup>
          </FormControl>
        </Paper>

        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Tùy chọn chung
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.autoDownload}
                  onChange={handleSwitchChange('autoDownload')}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="body1">Tự động tải xuống</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Tự động tải xuống file sau khi xử lý hoàn tất
                  </Typography>
                </Box>
              }
            />
            <Divider />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.preserveFormatting}
                  onChange={handleSwitchChange('preserveFormatting')}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="body1">Giữ nguyên định dạng</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Giữ nguyên định dạng và kiểu dáng của bảng
                  </Typography>
                </Box>
              }
            />
            <Divider />
            <FormControlLabel
              control={
                <Switch
                  checked={settings.includeMetadata}
                  onChange={handleSwitchChange('includeMetadata')}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="body1">Bao gồm metadata</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Thêm thông tin metadata vào file xuất
                  </Typography>
                </Box>
              }
            />
          </Box>
        </Paper>

        <Button
          variant="contained"
          size="large"
          onClick={handleSave}
          sx={{ minWidth: 200 }}
        >
          Lưu cài đặt
        </Button>

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

export default SettingsPage; 