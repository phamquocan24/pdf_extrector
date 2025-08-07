# Hướng dẫn cấu hình Email cho Reset Password

## Cấu hình biến môi trường

Thêm các biến sau vào file `.env` của bạn:

```env
# Email Configuration (Gmail)
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-gmail-app-password
EMAIL_FROM=PDF Extractor <noreply@pdfextractor.com>

# Frontend URL (for reset password links)
FRONTEND_URL=http://localhost:5173
```

## Cấu hình Gmail App Password

1. **Bật 2-Factor Authentication**:
   - Đăng nhập vào tài khoản Google
   - Vào Google Account settings
   - Chọn "Security" → "2-Step Verification"
   - Bật 2FA nếu chưa có

2. **Tạo App Password**:
   - Trong phần Security, chọn "App passwords"
   - Chọn "Mail" và thiết bị của bạn
   - Copy mật khẩu được tạo ra

3. **Cập nhật .env**:
   ```env
   EMAIL_USER=youremail@gmail.com
   EMAIL_PASS=generated-app-password
   ```

## Cấu hình email service khác

Nếu sử dụng email service khác, cập nhật file `backend/services/emailService.js`:

### Outlook/Hotmail:
```javascript
const transporter = nodemailer.createTransporter({
  service: 'hotmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS
  }
});
```

### SMTP tùy chỉnh:
```javascript
const transporter = nodemailer.createTransporter({
  host: 'your-smtp-host.com',
  port: 587,
  secure: false,
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS
  }
});
```

## Test email

Để test email service, bạn có thể sử dụng:

1. **Mailtrap** (cho development):
   ```javascript
   const transporter = nodemailer.createTransporter({
     host: "smtp.mailtrap.io",
     port: 2525,
     auth: {
       user: "your-mailtrap-user",
       pass: "your-mailtrap-pass"
     }
   });
   ```

2. **Ethereal Email** (cho testing):
   ```javascript
   const testAccount = await nodemailer.createTestAccount();
   const transporter = nodemailer.createTransporter({
     host: 'smtp.ethereal.email',
     port: 587,
     secure: false,
     auth: {
       user: testAccount.user,
       pass: testAccount.pass,
     },
   });
   ```

## Troubleshooting

### Lỗi "Invalid login":
- Kiểm tra email và password
- Đảm bảo đã bật 2FA và tạo App Password
- Kiểm tra "Less secure app access" (không khuyến nghị)

### Lỗi "Connection timeout":
- Kiểm tra firewall/network
- Thử port khác (465 cho SSL)

### Email không được gửi:
- Kiểm tra SPAM folder
- Kiểm tra daily sending limits
- Verify domain nếu sử dụng custom domain

## Security Notes

- Không commit file `.env` vào git
- Sử dụng App Password thay vì mật khẩu chính
- Rotate passwords định kỳ
- Monitor email sending logs