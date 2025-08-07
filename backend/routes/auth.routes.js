const express = require('express');
const router = express.Router();
const authController = require('../controllers/auth.controller');

// Auth routes
router.post('/login', authController.login);
router.post('/register', authController.register);
router.post('/google', authController.googleLogin);

// Password reset routes
router.post('/forgot-password', authController.forgotPassword);
router.post('/reset-password/:token', authController.resetPassword);
router.get('/verify-reset-token/:token', authController.verifyResetToken);

module.exports = router; 