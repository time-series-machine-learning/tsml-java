package tsml.data_containers.ts_fileIO;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import tsml.data_containers.TimeSeriesInstances;
import utilities.generic_storage.Pair;

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

    private HashMap<String, String> variables = new HashMap<>();

    private final StreamTokenizer m_Tokenizer;
    private int m_Lines;

    TimeSeriesInstances m_data;
    private String description;
    private String problemName;
    private boolean univariate = false;
    private boolean missing = false;
    private boolean timeStamps = false;
    private boolean classLabel;
    private List<String> classLabels;

    private List<List<List<Double>>> raw_data;

    private List<Double> raw_labels;

    public TSReader(Reader reader) throws IOException {
        m_Tokenizer = new StreamTokenizer(reader);
        initTokenizer();

        readHeader();

        CreateTimeSeriesInstances();
    }

    private void CreateTimeSeriesInstances() throws IOException {
        raw_data = new ArrayList<>();
        raw_labels = new ArrayList<>();

        // read each line and extract a data Instance
        Pair<List<List<Double>>, Double> multi_series_and_label;
        // extract the multivariate series, and the possible label.
        while ((multi_series_and_label = readMultivariateInstance()) != null) {
            raw_data.add(multi_series_and_label.var1);
            raw_labels.add(multi_series_and_label.var2);
        }

        // create timeseries instances object.
        m_data = new TimeSeriesInstances(raw_data, classLabels.toArray(new String[classLabels.size()]), raw_labels);
        m_data.setProblemName(problemName);
//        m_data.setHasTimeStamps(timeStamps); // todo this has been temp removed, should be computed from the data
        m_data.setDescription(description);
    }

    public TimeSeriesInstances GetInstances() {
        return m_data;
    }

    private Pair<List<List<Double>>, Double> readMultivariateInstance() throws IOException {
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            return null;
        }

        List<List<Double>> multi_timeSeries = new ArrayList<>();
        String classValue = "";

        ArrayList<Double> timeSeries = new ArrayList<>();
        do {
            // this means we're about to get the class value
            if (m_Tokenizer.ttype == ':' && classLabel) {
                // add the current time series to the list.
                multi_timeSeries.add(timeSeries);
                timeSeries = new ArrayList<>();
            } else {
                timeSeries.add(m_Tokenizer.sval == "?" ? Double.NaN : m_Tokenizer.nval);
                classValue = m_Tokenizer.sval == null ? "" + m_Tokenizer.nval : m_Tokenizer.sval; // the last value to
                                                                                                  // be tokenized should
                                                                                                  // be the class value.
                                                                                                  // can be in string or
                                                                                                  // number format so
                                                                                                  // check both.
            }
            m_Tokenizer.nextToken();
        } while (m_Tokenizer.ttype != StreamTokenizer.TT_EOL);

        // don't add the last series to the list, instead extract the first element and
        // figure out what the class value is.
        double classVal = classLabel ? (double) this.classLabels.indexOf(classValue) : -1.0;
        return new Pair<>(multi_timeSeries, classVal);
    }

    private void initTokenizer() {
        // Setup the tokenizer to read the stream.
        m_Tokenizer.resetSyntax();

        //// ignore 0-9 chars
        m_Tokenizer.wordChars(' ' + 1, '\u00FF');

        m_Tokenizer.parseNumbers();

        // setup the white space tokens
        m_Tokenizer.whitespaceChars(' ', ' ');
        m_Tokenizer.whitespaceChars(',', ',');

        // if we encounter a colon it means we need to start a new line? or it means a
        // new multivariate instance.
        m_Tokenizer.ordinaryChar(':');

        // setup the comment char
        m_Tokenizer.commentChar('#');

        // end of line is a significant token. it means the end of an instance.
        m_Tokenizer.eolIsSignificant(true);
    }

    // this function reads upto the @data bit in the file.
    protected void readHeader() throws IOException {
        // first token should be @problem name. as we skip whitespace and comments.

        // this gets the token there may be weirdness at the front of the file.
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        }

        do {

            String token = m_Tokenizer.sval;

            if (token.equalsIgnoreCase(CLASS_LABEL)) {
                ExtractClassLabels();
            } else {
                variables.put(token, ExtractVariable(token));
            }

            getNextToken();

        } while (!m_Tokenizer.sval.equalsIgnoreCase(DATA));

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

        // clear out last tokens.
        getLastToken(false);
    }

    private void ExtractClassLabels() throws IOException {
        classLabels = new ArrayList<>();
        getNextToken();
        classLabel = Boolean.parseBoolean(m_Tokenizer.sval);

        if (!classLabel) {
            getLastToken(false);
            return;
        }

        getNextToken();
        // now read all the class values until we reach the EOL
        do {
            classLabels.add(m_Tokenizer.sval == null ? "" + m_Tokenizer.nval : m_Tokenizer.sval);
            m_Tokenizer.nextToken();
        } while (m_Tokenizer.ttype != StreamTokenizer.TT_EOL);
    }

    private String ExtractVariable(String VARIABLE) throws IOException {
        // check if the current token matches the hardcoded value for @types e.g.
        // @problemName etc.
        getNextToken();
        String value = m_Tokenizer.sval;
        getLastToken(false);
        return value;
    }

    /**
     * Gets next token, skipping empty lines.
     *
     * @throws IOException if reading the next token fails
     */
    protected void getFirstToken() throws IOException {
        while (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {}
        ;
        // this handles quotations single and double/
        if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
            m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
            // this handles ? in the file.
        } else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD) && (m_Tokenizer.sval.equals("?"))) {
            m_Tokenizer.ttype = '?';
        }
    }

    /**
     * Gets next token, checking for a premature and of line.
     *
     * @throws IOException if it finds a premature end of line
     */
    protected void getNextToken() throws IOException {
        if (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
            errorMessage("premature end of line");
        }
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        } else if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
            m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
        } else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD) && (m_Tokenizer.sval.equals("?"))) {
            m_Tokenizer.ttype = '?';
        }
    }

    /**
     * Gets token and checks if its end of line.
     *
     * @param endOfFileOk whether EOF is OK
     * @throws IOException if it doesn't find an end of line
     */
    protected void getLastToken(boolean endOfFileOk) throws IOException {
        if ((m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL)
                && ((m_Tokenizer.ttype != StreamTokenizer.TT_EOF) || !endOfFileOk)) {
            errorMessage("end of line expected");
        }
    }

    /**
     * Throws error message with line number and last token read.
     *
     * @param msg the error message to be thrown
     * @throws IOException containing the error message
     */
    protected void errorMessage(String msg) throws IOException {
        String str = msg + ", read " + m_Tokenizer.toString();
        if (m_Lines > 0) {
            int line = Integer.parseInt(str.replaceAll(".* line ", ""));
            str = str.replaceAll(" line .*", " line " + (m_Lines + line - 1));
        }
        throw new IOException(str);
    }

    public static void main(String[] args) throws IOException {

        String local_path = "D:\\Work\\Data\\Univariate_ts\\";
        String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";

        String dataset = "ArrowHead";
        String filepath = local_path + dataset + "\\" + dataset;
        File f = new File(filepath + "_TRAIN" + ".ts");
        long time = System.nanoTime();
        TSReader ts_reader = new TSReader(new FileReader(f));
        System.out.println("after: " + (System.nanoTime() - time));

        TimeSeriesInstances train_data = ts_reader.GetInstances();

        System.out.println(train_data.toString());


        String dataset_multi = "FaceDetection";
        String filepath_multi = m_local_path + dataset_multi + "\\" + dataset_multi;
        File f1 = new File(filepath_multi + "_TRAIN" + ".ts");
        System.out.println(f1);
        time = System.nanoTime();
        TSReader ts_reader_multi = new TSReader(new FileReader(f1));
        TimeSeriesInstances train_data_multi = ts_reader_multi.GetInstances();
        // System.out.println(train_data_multi);
        System.out.println("after: " + (System.nanoTime() - time));

        System.out.println("Min: " + train_data_multi.getMinLength());
        System.out.println("Max: " + train_data_multi.getMaxLength());
    }
}
