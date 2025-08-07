import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  useTheme,
  Button,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Modal,
  Backdrop,
  Fade,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  TableChart as TableIcon,
  GridOn as CellIcon,
  GridOn as GridOnIcon,
  TextFields as TextIcon,
  Visibility as ViewIcon,
  Code as CodeIcon,
  ArrowBack as ArrowBackIcon,
  ContentCopy as ContentCopyIcon,
  Download as DownloadIcon,
  GetApp as GetAppIcon,
  Description as DescriptionIcon,
  TableView as TableViewIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
} from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';
import MainLayout from '../layouts/MainLayout';

const PreviewPage = () => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [extractedData, setExtractedData] = useState(null);
  const [fileName, setFileName] = useState('');
  const [downloadAnchorEl, setDownloadAnchorEl] = useState(null);
  const [imageModal, setImageModal] = useState({ open: false, src: '', title: '' });

  useEffect(() => {
    // Get data from route state or localStorage
    if (location.state && location.state.data) {
      console.log('Preview page received data:', location.state.data);
      // Log visualization data specifically
      location.state.data.forEach((table, index) => {
        console.log(`Table ${index + 1} visualizations:`, table.visualizations);
        if (table.visualizations?.table_detection_image) {
          console.log(`Table ${index + 1} has table detection image (length: ${table.visualizations.table_detection_image.length})`);
        }
        if (table.visualizations?.cell_segmentation_image) {
          console.log(`Table ${index + 1} has cell segmentation image (length: ${table.visualizations.cell_segmentation_image.length})`);
        }
      });
      setExtractedData(location.state.data);
      setFileName(location.state.fileName || 'Unknown File');
    } else {
      // If no data passed, redirect back to dashboard
      navigate('/dashboard');
    }
  }, [location, navigate]);

  const handleDownloadMenuOpen = (event) => {
    setDownloadAnchorEl(event.currentTarget);
  };

  const handleDownloadMenuClose = () => {
    setDownloadAnchorEl(null);
  };

  const handleImageClick = (src, title) => {
    setImageModal({ open: true, src, title });
  };

  const handleImageModalClose = () => {
    setImageModal({ open: false, src: '', title: '' });
  };

  const downloadFile = (content, filename, contentType) => {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    handleDownloadMenuClose();
  };

  const handleDownloadJSON = () => {
    const content = JSON.stringify(extractedData, null, 2);
    const filename = `${fileName.replace(/\.pdf$/i, '')}_extracted.json`;
    downloadFile(content, filename, 'application/json');
  };

  const handleDownloadCSV = () => {
    if (!extractedData || extractedData.length === 0) {
      alert('Không có dữ liệu bảng để xuất CSV');
      return;
    }
    
    const headers = ['STT', 'Trang', 'Bảng', 'Phương thức', 'Rows', 'Cols', 'Table Confidence', 'Rows Confidence', 'Cols Confidence', 'Structure Confidence', 'Quality Score', 'Min Confidence', 'Max Confidence', 'Cell Detection', 'Cells Count', 'Cells Confidence'];
    const csvContent = [
      headers.join(','),
      ...extractedData.map((table, index) => [
        index + 1,
        table.page || 'N/A',
        table.table || 'N/A',
        table.method || 'N/A',
        table.data?.length || 0,
        table.data?.[0]?.length || 0,
        table.table_detection?.confidence ? (table.table_detection.confidence * 100).toFixed(1) + '%' : 'N/A',
        'N/A', // Structure detection removed
        'N/A', // Structure detection removed  
        'N/A', // Structure detection removed
        table.table_detection?.confidence ? (table.table_detection.confidence * 100).toFixed(1) + '%' : 'N/A',
        'N/A', // Structure detection removed
        'N/A', // Structure detection removed
        table.cell_detection?.method || 'structure_based',
        table.cell_detection?.cells_detected || 0,
        table.cell_detection?.cells_confidence ? 
          (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length * 100).toFixed(1) + '%' : 'N/A'
      ].join(','))
    ].join('\n');
    
    const filename = `${fileName.replace(/\.pdf$/i, '')}_tables.csv`;
    downloadFile(csvContent, filename, 'text/csv');
  };

  const handleDownloadTXT = () => {
    if (!extractedData || extractedData.length === 0) {
      alert('Không có dữ liệu text để xuất TXT');
      return;
    }
    
    const textContent = extractedData
      .map((table, index) => 
        `Bảng ${table.table} - Trang ${table.page} (${table.method}):\n` +
        `Rows: ${table.data?.length || 0}, Cols: ${table.data?.[0]?.length || 0}\n` +
        `Confidence: ${table.table_detection?.confidence ? (table.table_detection.confidence * 100).toFixed(1) + '%' : 'N/A'}\n\n` +
        (table.data && table.data.length > 0 ? 
          table.data.map((row, rowIndex) => `Row ${rowIndex + 1}: [${row.join(', ')}]`).join('\n') :
          'Không có dữ liệu') + '\n'
      )
      .join('\n---\n\n');
    
    const filename = `${fileName.replace(/\.pdf$/i, '')}_text.txt`;
    downloadFile(textContent, filename, 'text/plain');
  };

  if (!extractedData) {
    return (
      <MainLayout>
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6">Không có dữ liệu để hiển thị</Typography>
        </Box>
      </MainLayout>
    );
  }

  const renderTableRegions = () => {
    if (!extractedData || extractedData.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary">
          Không tìm thấy vùng bảng nào
        </Typography>
      );
    }

    return extractedData.map((table, index) => (
      <Accordion key={index} sx={{ mb: 1 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <TableIcon color="primary" />
            <Typography variant="subtitle2">
              Bảng {table.table} - Trang {table.page}
            </Typography>
            {table.table_detection && (
              <Chip 
                label={`${(table.table_detection.confidence * 100).toFixed(1)}%`}
                size="small"
                color={table.table_detection.confidence > 0.7 ? "success" : "warning"}
                sx={{ ml: 1 }}
              />
            )}
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ p: 1 }}>
            {/* Table Detection Info */}
            <Typography variant="body2" color="primary" gutterBottom sx={{ fontWeight: 600 }}>
              📋 Table Detection: ✅ AI Model Detection
            </Typography>
            {table.table_detection && (
              <Box sx={{ mb: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 1 }}>
                  <Chip 
                    label={`Confidence: ${(table.table_detection.confidence * 100).toFixed(1)}%`} 
                    size="small" 
                    color={table.table_detection.confidence > 0.7 ? "success" : "warning"}
                  />
                  <Chip 
                    label={`Model: ${table.table_detection.model_used || 'N/A'}`} 
                    size="small" 
                    variant="outlined"
                  />
                </Box>
                {table.table_detection.bbox && (
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip label={`X: ${table.table_detection.bbox[0]}`} size="small" />
                    <Chip label={`Y: ${table.table_detection.bbox[1]}`} size="small" />
                    <Chip label={`W: ${table.table_detection.bbox[2] - table.table_detection.bbox[0]}`} size="small" />
                    <Chip label={`H: ${table.table_detection.bbox[3] - table.table_detection.bbox[1]}`} size="small" />
                  </Box>
                )}
              </Box>
            )}



            {/* Data Summary */}
            <Typography variant="body2" color="text.secondary" gutterBottom>
              📊 Dữ liệu trích xuất:
            </Typography>
            <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Typography variant="body2">
                Phương thức: <strong>3-Phase AI Workflow</strong>
              </Typography>
              <Typography variant="body2">
                Số hàng dữ liệu: <strong>{table.data?.length || 0}</strong>
              </Typography>
              <Typography variant="body2">
                Số cột dữ liệu: <strong>{table.data?.[0]?.length || 0}</strong>
              </Typography>
              <Typography variant="body2">
                Cells detected: <strong>{table.cell_detection?.cells_detected || 0}</strong>
              </Typography>
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>
    ));
  };

  const renderCellDetails = () => {
    // Get all table data combined
    const allTables = extractedData || [];
    
    if (allTables.length === 0) {
      return (
        <Box sx={{ p: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Không tìm thấy cell nào
          </Typography>
        </Box>
      );
    }

    return (
      <Box sx={{ p: 2 }}>
        <TableContainer 
          component={Paper} 
          sx={{ 
            maxHeight: 340,
            borderRadius: 1,
            boxShadow: 1,
          }}
        >
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>

                <TableCell sx={{ fontWeight: 600, bgcolor: 'grey.50' }}>Trang</TableCell>
                <TableCell sx={{ fontWeight: 600, bgcolor: 'grey.50' }}>Phương thức</TableCell>
                <TableCell sx={{ fontWeight: 600, bgcolor: 'grey.50' }}>Rows x Cols</TableCell>
                <TableCell sx={{ fontWeight: 600, bgcolor: 'grey.50' }}>Table Confidence</TableCell>

                <TableCell sx={{ fontWeight: 600, bgcolor: 'grey.50' }}>Cell Detection</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {allTables.map((table, index) => (
                <TableRow key={index} hover>

                  <TableCell>{table.page || 'N/A'}</TableCell>
                  <TableCell>
                    <Chip 
                      label="3-Phase AI"
                      size="small"
                      color="success"
                    />
                  </TableCell>
                  <TableCell>
                    {table.data?.length || 0} x {table.data?.[0]?.length || 0}
                  </TableCell>
                  <TableCell>
                    {table.table_detection?.confidence ? (
                      <Chip 
                        label={`${(table.table_detection.confidence * 100).toFixed(1)}%`}
                        size="small"
                        color={table.table_detection.confidence > 0.7 ? 'success' : table.table_detection.confidence > 0.5 ? 'warning' : 'error'}
                        sx={{ 
                          fontWeight: 'bold',
                          '& .MuiChip-label': {
                            fontSize: '0.75rem'
                          }
                        }}
                      />
                    ) : (
                      <Chip label="N/A" size="small" variant="outlined" />
                    )}
                  </TableCell>

                  <TableCell>
                    {table.cell_detection && table.cell_detection.cells_detected > 0 ? (
                      <Chip 
                        label={`${table.cell_detection.cells_detected} cells (${(table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length * 100).toFixed(1)}%)`}
                        size="small"
                        color={
                          (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length) > 0.7 ? 'success' : 
                          (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length) > 0.5 ? 'warning' : 'error'
                        }
                        sx={{ fontSize: '0.7rem', height: '20px' }}
                      />
                    ) : (
                      <Chip 
                        label="No cells"
                        size="small"
                        color="default"
                        variant="outlined"
                        sx={{ fontSize: '0.7rem', height: '20px' }}
                      />
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderTextCells = () => {
    // Flatten all table data to show text content
    const allTables = extractedData || [];
    let textCellIndex = 0;
    
    if (allTables.length === 0) {
      return (
        <Typography variant="body2" color="text.secondary">
          Không tìm thấy text cell nào
        </Typography>
      );
    }

    return allTables.map((table, tableIndex) => (
      <Accordion key={tableIndex} sx={{ mb: 1 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TextIcon color="primary" />
            <Typography variant="subtitle2">
              Bảng {table.table} - Trang {table.page}
            </Typography>
            <Chip 
              label={`${table.data?.length || 0} rows`} 
              size="small" 
              variant="outlined" 
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ p: 1 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Nội dung bảng (sample):
            </Typography>
            <Paper 
              sx={{ 
                p: 2, 
                bgcolor: 'grey.50', 
                border: '1px solid', 
                borderColor: 'grey.200',
                maxHeight: 200,
                overflow: 'auto'
              }}
            >
              {table.data && table.data.length > 0 ? (
                <Box>
                  {table.data.slice(0, 3).map((row, rowIndex) => (
                    <Typography key={rowIndex} variant="body2" sx={{ fontFamily: 'monospace', mb: 1 }}>
                      <strong>Row {rowIndex + 1}:</strong> [{row.join(', ')}]
                    </Typography>
                  ))}
                  {table.data.length > 3 && (
                    <Typography variant="caption" color="text.secondary">
                      ... và {table.data.length - 3} rows khác
                    </Typography>
                  )}
                </Box>
              ) : (
                <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                  Không có nội dung
                </Typography>
              )}
            </Paper>
            <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip 
                label="Phương thức: 3-Phase AI Workflow" 
                size="small" 
                color="success"
              />
              {table.table_detection && (
                <Chip 
                  label={`Table: ${(table.table_detection.confidence * 100).toFixed(1)}%`} 
                  size="small" 
                  color={table.table_detection.confidence > 0.7 ? 'success' : table.table_detection.confidence > 0.5 ? 'warning' : 'error'}
                />
              )}

              {table.cell_detection && table.cell_detection.cells_detected > 0 && (
                <Chip 
                  label={`Cells: ${table.cell_detection.cells_detected} (${(table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length * 100).toFixed(1)}%)`} 
                  size="small" 
                  color={
                    (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length) > 0.7 ? 'success' : 
                    (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length) > 0.5 ? 'warning' : 'error'
                  }
                  variant="outlined"
                />
              )}
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>
    ));
  };

  return (
    <MainLayout>
      <Box>
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <IconButton
              onClick={() => navigate('/dashboard')}
              sx={{
                bgcolor: 'primary.main',
                color: 'white',
                '&:hover': {
                  bgcolor: 'primary.dark',
                  transform: 'scale(1.05)',
                },
                transition: 'all 0.2s ease',
              }}
            >
              <ArrowBackIcon />
            </IconButton>
            <Typography 
              variant="h4" 
              component="h1"
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(45deg, #007BFF 30%, #28a745 90%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Chi tiết kết quả nhận dạng
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip 
                icon={<ViewIcon />}
                label={fileName}
                variant="outlined"
                color="primary"
                sx={{ fontSize: '0.9rem' }}
              />
              <Typography variant="body2" color="text.secondary">
                • {extractedData?.length || 0} bảng • {extractedData?.reduce((sum, table) => sum + (table.data?.length || 0), 0) || 0} rows • AI Detection: {extractedData?.filter(t => t.method === 'ai_model').length || 0}/{extractedData?.length || 0}
                {extractedData?.length > 0 && (
                  <>
                    {' • Avg Confidence: '}
                    <span style={{ 
                      color: extractedData.filter(t => t.table_detection?.confidence).reduce((avg, t) => avg + (t.table_detection.confidence * 100), 0) / extractedData.filter(t => t.table_detection?.confidence).length > 70 ? '#4caf50' : '#ff9800',
                      fontWeight: 'bold'
                    }}>
                      {extractedData.filter(t => t.table_detection?.confidence).length > 0 
                        ? (extractedData.filter(t => t.table_detection?.confidence).reduce((avg, t) => avg + (t.table_detection.confidence * 100), 0) / extractedData.filter(t => t.table_detection?.confidence).length).toFixed(1)
                        : '0'
                      }%
                    </span>
                  </>
                )}
              </Typography>
            </Box>
            
            <Button
              variant="contained"
              startIcon={<DownloadIcon />}
              endIcon={<KeyboardArrowDownIcon />}
              onClick={handleDownloadMenuOpen}
              sx={{
                bgcolor: 'success.main',
                '&:hover': {
                  bgcolor: 'success.dark',
                  transform: 'translateY(-1px)',
                },
                transition: 'all 0.2s ease',
                borderRadius: 2,
                px: 3,
              }}
            >
              Tải xuống
            </Button>
          </Box>
        </Box>

        {/* JSON Display - Horizontal Card */}
        <Card 
          sx={{ 
            mb: 3,
            transition: 'all 0.3s ease',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: theme.shadows[6],
            }
          }}
        >
          <CardHeader
            avatar={<CodeIcon color="primary" />}
            title="Dữ liệu JSON đầy đủ"
            subheader="Dữ liệu trích xuất hoàn chỉnh từ PDF"
            action={
              <Tooltip title="Sao chép JSON">
                <IconButton 
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(extractedData, null, 2));
                  }}
                  sx={{
                    '&:hover': {
                      bgcolor: 'primary.light',
                      color: 'white',
                    }
                  }}
                >
                  <ContentCopyIcon />
                </IconButton>
              </Tooltip>
            }
            sx={{ 
              bgcolor: 'grey.100',
              '& .MuiCardHeader-title': {
                fontWeight: 600
              }
            }}
          />
          <CardContent>
            <Paper
              sx={{
                p: 2,
                bgcolor: '#1e1e1e',
                border: '1px solid',
                borderColor: 'grey.300',
                maxHeight: 300,
                overflow: 'auto',
                borderRadius: 2,
                '&::-webkit-scrollbar': {
                  width: '8px',
                  height: '8px',
                },
                '&::-webkit-scrollbar-track': {
                  background: '#2a2a2a',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: '#555',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb:hover': {
                  background: '#777',
                },
              }}
            >
              <Typography
                component="pre"
                variant="body2"
                sx={{
                  fontFamily: '"Consolas", "Monaco", "Courier New", monospace',
                  fontSize: '0.75rem',
                  lineHeight: 1.4,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  color: '#f8f8f2',
                  '& .string': { color: '#e6db74' },
                  '& .number': { color: '#ae81ff' },
                  '& .boolean': { color: '#66d9ef' },
                  '& .null': { color: '#66d9ef' },
                  '& .key': { color: '#f92672' },
                }}
              >
                {JSON.stringify(extractedData, null, 2)}
              </Typography>
            </Paper>
          </CardContent>
        </Card>

        {/* Table Detection and Cell Segmentation Visualizations */}
        {extractedData && extractedData.length > 0 && (
          <Grid container spacing={3} sx={{ mb: 3 }}>
            {/* Table Detection Visualization */}
            <Grid item xs={12} md={6}>
              <Card 
                sx={{ 
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: theme.shadows[6],
                  }
                }}
              >
                <CardHeader
                  avatar={<TableIcon sx={{ color: 'white' }} />}
                  title="Ảnh bảng đã phát hiện"
                  subheader="Vùng bảng được AI model detect"
                  sx={{ 
                    bgcolor: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)',
                    background: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)',
                    color: 'white',
                    '& .MuiCardHeader-subheader': {
                      color: 'white',
                      opacity: 0.9
                    }
                  }}
                />
                <CardContent sx={{ p: 2 }}>
                  <Box
                    sx={{
                      maxHeight: 520, // Khoảng 4 ảnh (mỗi ảnh ~120px + margin)
                      overflowY: 'auto',
                      overflowX: 'hidden',
                      '&::-webkit-scrollbar': {
                        width: '8px',
                      },
                      '&::-webkit-scrollbar-track': {
                        background: '#f1f1f1',
                        borderRadius: '4px',
                      },
                      '&::-webkit-scrollbar-thumb': {
                        background: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)',
                        borderRadius: '4px',
                      },
                      '&::-webkit-scrollbar-thumb:hover': {
                        background: 'linear-gradient(135deg, #ee5a6f 0%, #ff6b6b 100%)',
                      },
                    }}
                  >
                    <Grid container spacing={2}>
                      {extractedData.map((table, index) => (
                        <Grid item xs={12} sm={6} md={12} lg={6} key={index}>
                          <Card variant="outlined" sx={{ borderRadius: 2 }}>
                            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                                Trang {table.page} - Bảng {table.table}
                              </Typography>
                              {table.visualizations?.table_detection_image ? (
                                <Box 
                                  component="img"
                                  src={table.visualizations.table_detection_image}
                                  alt={`Table detection page ${table.page} table ${table.table}`}
                                  onClick={() => handleImageClick(
                                    table.visualizations.table_detection_image,
                                    `Table Detection - Trang ${table.page}, Bảng ${table.table}`
                                  )}
                                  onError={(e) => {
                                    console.error('Failed to load table detection image:', e);
                                    e.target.style.display = 'none';
                                    e.target.nextSibling.style.display = 'flex';
                                  }}
                                  sx={{
                                    width: '100%',
                                    height: 'auto',
                                    maxHeight: 180,
                                    objectFit: 'contain',
                                    borderRadius: 1,
                                    border: '1px solid',
                                    borderColor: 'grey.300',
                                    cursor: 'pointer',
                                    transition: 'transform 0.2s ease',
                                    '&:hover': {
                                      transform: 'scale(1.02)',
                                      boxShadow: theme.shadows[4]
                                    }
                                  }}
                                />
                              ) : null}
                              {(!table.visualizations?.table_detection_image || table.visualizations?.error) && (
                                <Box 
                                  sx={{
                                    width: '100%',
                                    height: 180,
                                    bgcolor: 'grey.100',
                                    borderRadius: 1,
                                    display: 'flex',
                                    flexDirection: 'column',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    border: '1px solid',
                                    borderColor: 'grey.300',
                                    gap: 1
                                  }}
                                >
                                  <Typography variant="body2" color="text.secondary">
                                    {table.visualizations?.error ? 'Lỗi tạo ảnh visualization' : 'Không có ảnh visualization'}
                                  </Typography>
                                  {table.visualizations?.error && (
                                    <Typography variant="caption" color="error" sx={{ textAlign: 'center', px: 1 }}>
                                      {table.visualizations.error}
                                    </Typography>
                                  )}
                                </Box>
                              )}
                              <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                {table.table_detection?.confidence && (
                                  <Chip 
                                    label={`Confidence: ${(table.table_detection.confidence * 100).toFixed(1)}%`}
                                    size="small"
                                    color={table.table_detection.confidence > 0.7 ? 'success' : 'warning'}
                                  />
                                )}
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Cell Segmentation Visualization */}
            <Grid item xs={12} md={6}>
              <Card 
                sx={{ 
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: theme.shadows[6],
                  }
                }}
              >
                <CardHeader
                  avatar={<CellIcon sx={{ color: 'white' }} />}
                  title="Ảnh cell đã segment"
                  subheader="Từng cell được cắt ra và đánh số"
                  sx={{ 
                    bgcolor: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
                    background: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
                    color: 'white',
                    '& .MuiCardHeader-subheader': {
                      color: 'white',
                      opacity: 0.9
                    }
                  }}
                />
                <CardContent sx={{ p: 2 }}>
                  <Box
                    sx={{
                      maxHeight: 520, // Khoảng 4 ảnh (mỗi ảnh ~120px + margin)
                      overflowY: 'auto',
                      overflowX: 'hidden',
                      '&::-webkit-scrollbar': {
                        width: '8px',
                      },
                      '&::-webkit-scrollbar-track': {
                        background: '#f1f1f1',
                        borderRadius: '4px',
                      },
                      '&::-webkit-scrollbar-thumb': {
                        background: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
                        borderRadius: '4px',
                      },
                      '&::-webkit-scrollbar-thumb:hover': {
                        background: 'linear-gradient(135deg, #fed6e3 0%, #a8edea 100%)',
                      },
                    }}
                  >
                    <Grid container spacing={2}>
                      {extractedData.map((table, index) => (
                        <Grid item xs={12} sm={6} md={12} lg={6} key={index}>
                          <Card variant="outlined" sx={{ borderRadius: 2 }}>
                            <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                              <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
                                Trang {table.page} - Bảng {table.table}
                              </Typography>
                              {table.visualizations?.cell_segmentation_image ? (
                                <Box 
                                  component="img"
                                  src={table.visualizations.cell_segmentation_image}
                                  alt={`Cell segmentation page ${table.page} table ${table.table}`}
                                  onClick={() => handleImageClick(
                                    table.visualizations.cell_segmentation_image,
                                    `Cell Segmentation - Trang ${table.page}, Bảng ${table.table}`
                                  )}
                                  onError={(e) => {
                                    console.error('Failed to load cell segmentation image:', e);
                                    e.target.style.display = 'none';
                                    e.target.nextSibling.style.display = 'flex';
                                  }}
                                  sx={{
                                    width: '100%',
                                    height: 'auto',
                                    maxHeight: 180,
                                    objectFit: 'contain',
                                    borderRadius: 1,
                                    border: '1px solid',
                                    borderColor: 'grey.300',
                                    cursor: 'pointer',
                                    transition: 'transform 0.2s ease',
                                    '&:hover': {
                                      transform: 'scale(1.02)',
                                      boxShadow: theme.shadows[4]
                                    }
                                  }}
                                />
                              ) : null}
                              {(!table.visualizations?.cell_segmentation_image || table.visualizations?.error) && (
                                <Box 
                                  sx={{
                                    width: '100%',
                                    height: 180,
                                    bgcolor: 'grey.100',
                                    borderRadius: 1,
                                    display: 'flex',
                                    flexDirection: 'column',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    border: '1px solid',
                                    borderColor: 'grey.300',
                                    gap: 1
                                  }}
                                >
                                  <Typography variant="body2" color="text.secondary">
                                    {table.visualizations?.error ? 'Lỗi tạo ảnh cell visualization' : 'Không có ảnh cell visualization'}
                                  </Typography>
                                  {table.visualizations?.error && (
                                    <Typography variant="caption" color="error" sx={{ textAlign: 'center', px: 1 }}>
                                      {table.visualizations.error}
                                    </Typography>
                                  )}
                                </Box>
                              )}
                              <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                <Chip 
                                  label={`${table.cell_detection?.cells_detected || 0} cells`}
                                  size="small"
                                  color="info"
                                />
                                {table.cell_detection?.cells_confidence && table.cell_detection.cells_confidence.length > 0 && (
                                  <Chip 
                                    label={`Avg: ${(table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length * 100).toFixed(1)}%`}
                                    size="small"
                                    color={
                                      (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length) > 0.7 ? 'success' : 
                                      (table.cell_detection.cells_confidence.reduce((a, b) => a + b, 0) / table.cell_detection.cells_confidence.length) > 0.5 ? 'warning' : 'error'
                                    }
                                  />
                                )}
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Three Vertical Cards */}
        <Grid container spacing={3}>
          {/* Table Regions Card */}
          <Grid item xs={12} lg={4}>
            <Card 
              sx={{ 
                height: { xs: 'auto', lg: '600px' },
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[8],
                }
              }}
            >
              <CardHeader
                avatar={<TableIcon sx={{ color: 'white' }} />}
                title="Vùng bảng nhận dạng"
                subheader={`${extractedData?.length || 0} vùng bảng • ${extractedData?.reduce((sum, table) => sum + (table.cell_detection?.cells_detected || 0), 0) || 0} cells detected`}
                sx={{ 
                  bgcolor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  '& .MuiCardHeader-subheader': {
                    color: 'white',
                    opacity: 0.9
                  }
                }}
              />
              <Divider />
              <CardContent sx={{ 
                maxHeight: { xs: 300, lg: 460 }, 
                overflow: 'auto',
                '&::-webkit-scrollbar': {
                  width: '8px',
                },
                '&::-webkit-scrollbar-track': {
                  background: '#f1f1f1',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb:hover': {
                  background: 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
                },
              }}>
                {renderTableRegions()}
              </CardContent>
            </Card>
          </Grid>

          {/* Cell Details Card */}
          <Grid item xs={12} lg={4}>
            <Card 
              sx={{ 
                height: { xs: 'auto', lg: '600px' },
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[8],
                }
              }}
            >
              <CardHeader
                avatar={<CellIcon sx={{ color: 'white' }} />}
                title="Chi tiết các bảng"
                subheader={`${extractedData?.length || 0} bảng phát hiện`}
                sx={{ 
                  bgcolor: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                  background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                  color: 'white',
                  '& .MuiCardHeader-subheader': {
                    color: 'white',
                    opacity: 0.9
                  }
                }}
              />
              <Divider />
              <CardContent sx={{ p: 0, height: { xs: 300, lg: 460 }, overflow: 'hidden' }}>
                {renderCellDetails()}
              </CardContent>
            </Card>
          </Grid>

          {/* Text Cells Card */}
          <Grid item xs={12} lg={4}>
            <Card 
              sx={{ 
                height: { xs: 'auto', lg: '600px' },
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: theme.shadows[8],
                }
              }}
            >
              <CardHeader
                avatar={<TextIcon sx={{ color: 'white' }} />}
                title="Nội dung bảng"
                subheader={`${extractedData?.reduce((sum, table) => sum + (table.data?.length || 0), 0) || 0} rows tổng`}
                sx={{ 
                  bgcolor: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                  background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                  color: 'white',
                  '& .MuiCardHeader-subheader': {
                    color: 'white',
                    opacity: 0.9
                  }
                }}
              />
              <Divider />
              <CardContent sx={{ 
                maxHeight: { xs: 300, lg: 460 }, 
                overflow: 'auto',
                '&::-webkit-scrollbar': {
                  width: '8px',
                },
                '&::-webkit-scrollbar-track': {
                  background: '#f1f1f1',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb:hover': {
                  background: 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)',
                },
              }}>
                {renderTextCells()}
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Download Menu */}
        <Menu
          anchorEl={downloadAnchorEl}
          open={Boolean(downloadAnchorEl)}
          onClose={handleDownloadMenuClose}
          PaperProps={{
            sx: { 
              minWidth: 200,
              mt: 1,
              borderRadius: 2,
              boxShadow: theme.shadows[8],
            },
          }}
          transformOrigin={{ horizontal: 'right', vertical: 'top' }}
          anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        >
          <MenuItem onClick={handleDownloadJSON}>
            <ListItemIcon>
              <CodeIcon fontSize="small" color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="JSON File" 
              secondary="Dữ liệu đầy đủ"
            />
          </MenuItem>
          
          <MenuItem onClick={handleDownloadCSV}>
            <ListItemIcon>
              <TableViewIcon fontSize="small" color="secondary" />
            </ListItemIcon>
            <ListItemText 
              primary="CSV File" 
              secondary="Dữ liệu cells"
            />
          </MenuItem>
          
          <MenuItem onClick={handleDownloadTXT}>
            <ListItemIcon>
              <DescriptionIcon fontSize="small" color="info" />
            </ListItemIcon>
            <ListItemText 
              primary="TXT File" 
              secondary="Nội dung text"
            />
          </MenuItem>
        </Menu>

        {/* Image Modal */}
        <Modal
          open={imageModal.open}
          onClose={handleImageModalClose}
          closeAfterTransition
          BackdropComponent={Backdrop}
          BackdropProps={{
            timeout: 500,
            sx: { backgroundColor: 'rgba(0, 0, 0, 0.8)' }
          }}
        >
          <Fade in={imageModal.open}>
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                maxWidth: '95vw',
                maxHeight: '95vh',
                bgcolor: 'background.paper',
                borderRadius: 2,
                boxShadow: 24,
                p: 2,
                outline: 'none',
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
                  {imageModal.title}
                </Typography>
                <IconButton onClick={handleImageModalClose} sx={{ color: 'grey.500' }}>
                  <ArrowBackIcon />
                </IconButton>
              </Box>
              <Box 
                component="img"
                src={imageModal.src}
                alt={imageModal.title}
                sx={{
                  maxWidth: '100%',
                  maxHeight: '80vh',
                  objectFit: 'contain',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'grey.300'
                }}
              />
            </Box>
          </Fade>
        </Modal>
      </Box>
    </MainLayout>
  );
};

export default PreviewPage;