/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.data_containers.ts_fileIO;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

/**
 * File for reading sktime format data into TimeSeriesInstances object
 * 
 * @author Aaron Bostrom, pushed 22/4/2020
 */

public class TSReader {

    // need to change this to a map function.
    public static final String PROBLEM_NAME = "@problemName";
    public static final String TIME_STAMPS = "@timeStamps";
    public static final String CLASS_LABEL = "@classLabel";
    public static final String UNIVARIATE = "@univariate";
    public static final String MISSING = "@missing";
    public static final String DATA = "@data";

    private HashMap<String, String> variables;

    private final Scanner m_scanner;
    private String currentToken;
    private int m_Lines;


    private String description;
    private String problemName;
    private boolean univariate;
    private boolean missing;
    private boolean timeStamps;
    private boolean classLabel;
    private List<String> classLabels;

    TimeSeriesInstances m_data;
    private List<TimeSeriesInstance> raw_data;

    public TSReader(Reader reader) throws IOException {
        variables = new HashMap<>();
        //m_Tokenizer = new StreamTokenizer(reader);

        m_scanner = new Scanner(reader);
        m_scanner.useDelimiter("[\\s,]+");

        readHeader();

        System.out.println(variables);
        System.out.println(classLabels);

        CreateTimeSeriesInstances();
    }

    private void CreateTimeSeriesInstances() throws IOException {
        // read each line and extract a data Instance
        raw_data = new ArrayList<>();
        // extract the multivariate series, and the possible label.

        while (m_scanner.hasNextLine()) {
            String line = m_scanner.nextLine();
            Scanner lineScanner = new Scanner(line);
            lineScanner.useDelimiter("((?=[:,])|(?<=[:,]))");

            raw_data.add(readMultivariateInstance(lineScanner));

            lineScanner.close();
        }

        // create timeseries instances object.
        m_data = new TimeSeriesInstances(raw_data, classLabels.toArray(new String[classLabels.size()]));
        m_data.setProblemName(problemName);
//        m_data.setHasTimeStamps(timeStamps); // todo this has been temp removed, should be computed from the data
        m_data.setDescription(description);
    }

    public TimeSeriesInstances GetInstances() {
        return m_data;
    }

    private TimeSeriesInstance readMultivariateInstance(Scanner lineScanner) throws IOException {
        List<List<Double>> multi_timeSeries = new ArrayList<>();
        String classValue = "";

        ArrayList<Double> timeSeries = new ArrayList<>();
        while (lineScanner.hasNext()){
            getNextToken(lineScanner);
            // this means we're about to get the class value
            if (currentToken.equalsIgnoreCase(":") && classLabel) {
                
                // add the current time series to the list.
                multi_timeSeries.add(timeSeries);
                timeSeries = new ArrayList<>();
            } else {

                double val;

                try{
                    val = Double.parseDouble(currentToken);
                   
                }catch(NumberFormatException ex){
                    val = Double.NaN;
                }

                timeSeries.add(val);
                classValue = currentToken;
            }
        } 

        // don't add the last series to the list, instead extract the first element and
        // figure out what the class value is.
        int classVal = classLabel ?  classLabels.indexOf(classValue) : -1;

        return new TimeSeriesInstance(multi_timeSeries, classVal);
    }

    // this function reads upto the @data bit in the file.
    protected void readHeader() throws IOException {
        // first token should be @problem name. as we skip whitespace and comments.

        skipComments();
        getNextToken();

        do {           
            if (currentToken.equalsIgnoreCase(CLASS_LABEL)) {
                ExtractClassLabels();
            } else {
                variables.put(currentToken, getNextToken());    
                getNextToken(); 
            }

        } while (!currentToken.equalsIgnoreCase(DATA));
        
        // these are required.
        problemName = variables.get(PROBLEM_NAME);
        if (problemName == null) {
            errorMessage("keyword " + PROBLEM_NAME + " expected");
        }

        if (variables.get(UNIVARIATE) == null) {
            errorMessage("keyword " + UNIVARIATE + " expected");
        } else {
            univariate = Boolean.parseBoolean(variables.get(UNIVARIATE));
        }

        // set optionals.
        if (variables.get(MISSING) != null)
            missing = Boolean.parseBoolean(variables.get(MISSING));
        if (variables.get(TIME_STAMPS) != null)
            timeStamps = Boolean.parseBoolean(variables.get(TIME_STAMPS));


        m_scanner.nextLine(); //clear our this bit.
    }

    private void ExtractClassLabels() throws IOException {
        classLabels = new ArrayList<>();

        if(m_scanner.hasNextBoolean())
            classLabel = m_scanner.nextBoolean();

        if (!classLabel)
            return;

        while (!getNextToken().contains("@")){
            classLabels.add(currentToken);
        }
        
    }

    protected void skipComments(){
        while (m_scanner.findInLine("\\s*\\#.*") != null) {m_scanner.nextLine();}
    }

    /**
     * Gets next token, checking for a premature and of line.
     *
     * @throws IOException if it finds a premature end of line
     */
    protected String getNextToken() throws IOException {
        return getNextToken(m_scanner);
    }

    protected String getNextToken(Scanner scanner) throws IOException {
        currentToken = scanner.next();
        //recurse until we find a valid token. 
        if (currentToken.equalsIgnoreCase(","))
            getNextToken(scanner);

        //System.out.println("t: "+currentToken);
        return currentToken;
    }

    /**
     * Throws error message with line number and last token read.
     *
     * @param msg the error message to be thrown
     * @throws IOException containing the error message
     */
    protected void errorMessage(String msg) throws IOException {
        String str = msg + ", read " + currentToken;
        if (m_Lines > 0) {
            int line = Integer.parseInt(str.replaceAll(".* line ", ""));
            str = str.replaceAll(" line .*", " line " + (m_Lines + line - 1));
        }
        throw new IOException(str);
    }

    public static void main(String[] args) throws IOException {

        // String local_path = "D:\\Work\\Data\\Univariate_ts\\";
        // String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";

        String local_path = "Z:\\ArchiveData\\Univariate_ts\\";
        String m_local_path = "Z:\\ArchiveData\\Multivariate_ts\\";

        String[] paths = {/*local_path,*/ m_local_path};

        for (String path : paths){
            File dir = new File(path);
            for (File file : dir.listFiles()){
                String filepath = path + file.getName() + "\\" + file.getName();
                File f = new File(filepath + "_TRAIN" + ".ts");
                long time = System.nanoTime();
                TSReader ts_reader = new TSReader(new FileReader(f));
                System.out.println("after: " + (System.nanoTime() - time));
            }
        }
    }
}
