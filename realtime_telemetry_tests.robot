*** Settings ***
Documentation     Test suite for real-time robotic telemetry analysis
Library           realtime_telemetry_analyzer.RoboticTelemetryAnalyzer
Library           OperatingSystem

*** Test Cases ***
Run Real-Time Telemetry Analysis
    [Documentation]    Test real-time telemetry analysis
    [Timeout]    2 minutes
    
    Start Real-Time Analysis
    
    # Monitor analysis for some time
    FOR    ${index}    IN RANGE    10
        ${results}=    Get Latest Analysis Results
        Run Keyword If    ${results}    Log Results    ${results}
        Sleep    5s
    END
    
    Stop Real-Time Analysis

*** Keywords ***
Log Results
    [Arguments]    ${results}
    Log    Timestamp: ${results}[timestamp]
    Log    Risk Level: ${results}[latest_risk_level]
    Log    Risk Score: ${results}[latest_risk_score]
    Log    High Risk Count: ${results}[high_risk_count]
    Log    Total Count: ${results}[total_count]