@echo off
echo Starting all services for PDF Extractor...

:: Start Python service
echo Starting Python service on port 8001...
cd backend\python_service
start "Python Service" cmd /k "python app.py"
cd ..\..

:: Start Node.js backend on port 8080
echo Starting Node.js backend on port 8080...
cd backend
start "Node Backend" cmd /k "npm start"
cd ..

:: Start Frontend on port 5173 
echo Starting Frontend on port 5173...
cd frontend
start "Frontend" cmd /k "npm run dev"
cd ..

echo All services started! 
echo - Python Service: http://localhost:8001
echo - Node Backend: http://localhost:8080  
echo - Frontend: http://localhost:5173

pause
