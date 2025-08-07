const nodemailer = require('nodemailer');

// Create transporter (using Gmail as example)
const createTransporter = () => {
  return nodemailer.createTransporter({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER || 'your-email@gmail.com',
      pass: process.env.EMAIL_PASS || 'your-app-password'
    }
  });
};

// Send reset password email
const sendResetPasswordEmail = async (email, resetToken, userName) => {
  try {
    const transporter = createTransporter();
    
    const resetUrl = `${process.env.FRONTEND_URL || 'http://localhost:5173'}/reset-password/${resetToken}`;
    
    const mailOptions = {
      from: process.env.EMAIL_FROM || 'PDF Extractor <noreply@pdfextractor.com>',
      to: email,
      subject: 'Yêu cầu đặt lại mật khẩu - PDF Extractor',
      html: `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Đặt lại mật khẩu</title>
          <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #007BFF; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
            .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
            .button { display: inline-block; background: #007BFF; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
            .footer { text-align: center; margin-top: 20px; color: #666; font-size: 12px; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1>PDF Extractor</h1>
              <h2>Đặt lại mật khẩu</h2>
            </div>
            
            <div class="content">
              <p>Xin chào <strong>${userName}</strong>,</p>
              
              <p>Chúng tôi nhận được yêu cầu đặt lại mật khẩu cho tài khoản của bạn tại PDF Extractor.</p>
              
              <p>Để đặt lại mật khẩu, vui lòng click vào nút bên dưới:</p>
              
              <div style="text-align: center;">
                <a href="${resetUrl}" class="button">Đặt lại mật khẩu</a>
              </div>
              
              <p>Hoặc copy và dán link sau vào trình duyệt:</p>
              <p style="word-break: break-all; background: #f1f1f1; padding: 10px; border-radius: 5px;">
                ${resetUrl}
              </p>
              
              <div class="warning">
                <strong>⚠️ Lưu ý quan trọng:</strong>
                <ul>
                  <li>Link này chỉ có hiệu lực trong <strong>1 giờ</strong></li>
                  <li>Nếu bạn không yêu cầu đặt lại mật khẩu, vui lòng bỏ qua email này</li>
                  <li>Để bảo mật, không chia sẻ link này với ai khác</li>
                </ul>
              </div>
              
              <p>Nếu bạn gặp vấn đề, vui lòng liên hệ đội ngũ hỗ trợ của chúng tôi.</p>
              
              <p>Trân trọng,<br><strong>Đội ngũ PDF Extractor</strong></p>
            </div>
            
            <div class="footer">
              <p>Email này được gửi tự động, vui lòng không reply.</p>
              <p>&copy; 2025 PDF Extractor. All rights reserved.</p>
            </div>
          </div>
        </body>
        </html>
      `
    };
    
    const result = await transporter.sendMail(mailOptions);
    console.log('Reset password email sent successfully:', result.messageId);
    return { success: true, messageId: result.messageId };
    
  } catch (error) {
    console.error('Error sending reset password email:', error);
    throw new Error('Không thể gửi email. Vui lòng thử lại sau.');
  }
};

// Send email confirmation after password reset
const sendPasswordResetConfirmation = async (email, userName) => {
  try {
    const transporter = createTransporter();
    
    const mailOptions = {
      from: process.env.EMAIL_FROM || 'PDF Extractor <noreply@pdfextractor.com>',
      to: email,
      subject: 'Mật khẩu đã được đặt lại thành công - PDF Extractor',
      html: `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Mật khẩu đã được đặt lại</title>
          <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            .header { background: #28a745; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
            .content { background: #f9f9f9; padding: 30px; border-radius: 0 0 8px 8px; }
            .success { background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .footer { text-align: center; margin-top: 20px; color: #666; font-size: 12px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1>PDF Extractor</h1>
              <h2>✅ Mật khẩu đã được đặt lại</h2>
            </div>
            
            <div class="content">
              <p>Xin chào <strong>${userName}</strong>,</p>
              
              <div class="success">
                <strong>🎉 Thành công!</strong><br>
                Mật khẩu của bạn đã được đặt lại thành công vào lúc ${new Date().toLocaleString('vi-VN')}.
              </div>
              
              <p>Bạn có thể đăng nhập vào tài khoản bằng mật khẩu mới.</p>
              
              <p><strong>Một số lưu ý bảo mật:</strong></p>
              <ul>
                <li>Không chia sẻ mật khẩu với bất kỳ ai</li>
                <li>Sử dụng mật khẩu mạnh, khó đoán</li>
                <li>Đăng xuất khỏi các thiết bị không tin cậy</li>
              </ul>
              
              <p>Nếu bạn không thực hiện thao tác này, vui lòng liên hệ ngay với đội ngũ hỗ trợ.</p>
              
              <p>Trân trọng,<br><strong>Đội ngũ PDF Extractor</strong></p>
            </div>
            
            <div class="footer">
              <p>Email này được gửi tự động, vui lòng không reply.</p>
              <p>&copy; 2025 PDF Extractor. All rights reserved.</p>
            </div>
          </div>
        </body>
        </html>
      `
    };
    
    const result = await transporter.sendMail(mailOptions);
    console.log('Password reset confirmation email sent successfully:', result.messageId);
    return { success: true, messageId: result.messageId };
    
  } catch (error) {
    console.error('Error sending confirmation email:', error);
    // Don't throw error for confirmation email failure
    return { success: false, error: error.message };
  }
};

module.exports = {
  sendResetPasswordEmail,
  sendPasswordResetConfirmation
};