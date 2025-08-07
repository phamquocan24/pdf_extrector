import React from 'react';
import { Box, Typography, Paper, Avatar, Grid } from '@mui/material';
import { useAuth } from '../contexts/AuthContext'; // Import useAuth

const ProfilePage = () => {
  const { user } = useAuth(); // Get user from context

  if (!user) {
    return (
      <Box>
        <Typography>Loading user profile...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        User Profile
      </Typography>
      <Paper sx={{ p: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item>
            <Avatar
              src={user.picture}
              sx={{ width: 80, height: 80, bgcolor: 'primary.main' }}
            >
              {user.name?.charAt(0)}
            </Avatar>
          </Grid>
          <Grid item>
            <Typography variant="h5">{user.name}</Typography>
            <Typography variant="body1" color="text.secondary">{user.email}</Typography>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default ProfilePage; 