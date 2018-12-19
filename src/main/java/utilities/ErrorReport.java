package utilities;

/**
 * Noddy little class for situations where multiple exceptions may be thrown during some 
 * action and you want to collect/report them all instead of just the first to help with debugging
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ErrorReport {
    private boolean anyErrors; 
    private String errorLog;

    public ErrorReport(String msgHeader) {
        errorLog = msgHeader;
    }

    public void log(String errorMsg) {
        anyErrors = true;
        errorLog += errorMsg;
    }

    public void throwIfErrors() throws Exception {
        if (anyErrors)
            throw new Exception(errorLog);
    }

    public boolean isEmpty() { return !anyErrors; };
    public String getLog() { return errorLog; };
    public void setLog(String newLog) { errorLog = newLog; };
}