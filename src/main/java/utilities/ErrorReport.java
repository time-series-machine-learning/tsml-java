/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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