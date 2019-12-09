package utilities;

import weka.core.*;

import java.io.*;
import java.util.ArrayList;
import java.util.function.IntConsumer;

public class TSReader {

    public static final String PROBLEM_NAME = "@problemName";
    public static final String TIME_STAMPS = "@timeStamps";
    public static final String CLASS_LABEL = "@classLabel";
    public static final String UNIVARIATE = "@univariate";
    public static final String DATA = "@data";


    private final StreamTokenizer m_Tokenizer;
    private int m_Lines;

    Instances m_data;
    private String problemName;
    private boolean timeStamps;
    private boolean univariate = false;
    private boolean classLabel;
    private ArrayList<String> classLabels;
    private ArrayList<Attribute> attList;

    private ArrayList<ArrayList<Double>> raw_data;

    public TSReader(Reader reader) throws IOException{
        m_Tokenizer = new StreamTokenizer(reader);
        initTokenizer();

        readHeader();

        raw_data = new ArrayList<>();
        //read each line and extract a data Instance
        ArrayList<Double> series;
        while((series = readInstance()) != null){
            raw_data.add(series);
        }


        //go through all the raw data, and find the longest row.
        int max_length = raw_data.stream().mapToInt(ArrayList::size).max().getAsInt();
        System.out.println(max_length);

        //max length is one less if we have chucked the class value on the end.
        if(classLabel)
            max_length--;


        // create attribute list
        attList = new ArrayList<>();
        for(int i = 0; i < max_length; i++){
            attList.add(new Attribute("att"+(i+1), i));
        }
        //have to cast the null to arraylist type (this is taken from WEKAS OWN CODE.) to force it to use the right method.
        if(classLabel)
            attList.add(new Attribute("classVal", classLabels, classLabels.size()));

        m_data = new Instances(problemName, attList, raw_data.size());

        for(ArrayList<Double> timeSeries : raw_data){
            //add all the time series values.
            Instance inst = new DenseInstance(max_length+1);
            for(int a = 0; a < timeSeries.size(); a++){
                inst.setValue(a, timeSeries.get(a));
            }
            //only add if we have a classLabel
            //get the value from the end of the current time series, and put it at the end of the attribute list.
            if(classLabel)
                inst.setValue(max_length, timeSeries.get(timeSeries.size()-1));

            m_data.add(inst);
        }
    }

    public Instances GetInstances(){
        return m_data;
    }

    private ArrayList<Double> readInstance() throws IOException{
        if(!univariate){
            return readMultivariateInstance();
        }
        else{
            return readUnivariateInstance();
        }
    }

    private  ArrayList<Double> readMultivariateInstance() throws IOException {
        return null;
    }

    private  ArrayList<Double> readUnivariateInstance() throws IOException {
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            return null;
        }

        ArrayList<Double> timeSeries = new ArrayList<>();
        String classValue = null;

        //read the tokens, and if we hit a : then we need to do something clever.
        boolean bFoundColon = false;
        do{
            //this means we're about to get the class value
            if(m_Tokenizer.ttype == ':' && classLabel){
                bFoundColon = true;
            }
            else{
                if(bFoundColon){
                    classValue = m_Tokenizer.sval;
                    bFoundColon = false;

                    //get the index of the class value and cast it to double.
                    timeSeries.add((double)this.classLabels.indexOf(classValue));
                }
                else{
                    timeSeries.add(Double.valueOf(m_Tokenizer.sval));
                }
            }
            m_Tokenizer.nextToken();
        } while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);

        return timeSeries;
    }

    private void initTokenizer() {
        //Setup the tokenizer to read the stream.
        m_Tokenizer.resetSyntax();

        //chars are the chars.
        m_Tokenizer.wordChars(' '+1,'\u00FF');

        //setup the white space tokens
        m_Tokenizer.whitespaceChars(' ', ' ');
        m_Tokenizer.whitespaceChars(',',',');

        //if we encounter a colon it means we need to start a new line? or it means a new multivariate instance.
        m_Tokenizer.ordinaryChar(':');

        //setup the comment char
        m_Tokenizer.commentChar('#');

        //end of line is a significant token. it means the end of an instance.
        m_Tokenizer.eolIsSignificant(true);
    }

    //this function reads upto the @data bit in the file.
    protected void readHeader() throws IOException {
        //first token should be @problem name. as we skip whitespace and comments.
        problemName = ExtractVariable(PROBLEM_NAME);
        timeStamps = Boolean.parseBoolean(ExtractVariable(TIME_STAMPS));
        ExtractClassLabels();
        univariate = Boolean.parseBoolean(ExtractVariable(UNIVARIATE));

        //find @data.
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        }

        //if we've found @data then clear out any tokens up to the first point of data.
        if(m_Tokenizer.sval.equalsIgnoreCase(DATA)){
            getLastToken(false);
        }
    }

    private void ExtractClassLabels() throws IOException {
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        }

        classLabels = new ArrayList<>();
        if(m_Tokenizer.sval.equalsIgnoreCase(CLASS_LABEL)){
            getNextToken();
            classLabel = Boolean.parseBoolean(m_Tokenizer.sval);

            if(!classLabel) {
                getLastToken(false);
                return;
            }

            getNextToken();
            //now read all the class values until we reach the EOL
            do{
                classLabels.add(m_Tokenizer.sval);
                m_Tokenizer.nextToken();
            }while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);
        } else {
            errorMessage("keyword " + CLASS_LABEL + " expected");
        }
    }

    private String ExtractVariable(String VARIABLE) throws IOException {
        //this gets the token there may be weirdness at the front of the file.
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        }

        //check if the current token matches the hardcoded value for @types e.g. @problemName etc.
        String value = null;
        if (m_Tokenizer.sval.equalsIgnoreCase(VARIABLE)) {
            getNextToken();
            value = m_Tokenizer.sval;
            getLastToken(false);
        } else {
            errorMessage("keyword " + VARIABLE + " expected");
        }

        return value;
    }

    /**
     * Gets next token, skipping empty lines.
     *
     * @throws IOException 	if reading the next token fails
     */
    protected void getFirstToken() throws IOException {
        while (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {};
        //this handles quotations single and double/
        if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
            m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
            // this handles ? in the file.
        } else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD) && (m_Tokenizer.sval.equals("?"))){
            m_Tokenizer.ttype = '?';
        }
    }


    /**
     * Gets next token, checking for a premature and of line.
     *
     * @throws IOException 	if it finds a premature end of line
     */
    protected void getNextToken() throws IOException {
        if (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
            errorMessage("premature end of line");
        }
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        } else if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
            m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
        } else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD) && (m_Tokenizer.sval.equals("?"))){
            m_Tokenizer.ttype = '?';
        }
    }

    /**
     * Gets token and checks if its end of line.
     *
     * @param endOfFileOk 	whether EOF is OK
     * @throws IOException 	if it doesn't find an end of line
     */
    protected void getLastToken(boolean endOfFileOk) throws IOException {
        if ((m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL) && ((m_Tokenizer.ttype != StreamTokenizer.TT_EOF) || !endOfFileOk)) {
            errorMessage("end of line expected");
        }
    }


    /**
     * Throws error message with line number and last token read.
     *
     * @param msg 		the error message to be thrown
     * @throws IOException 	containing the error message
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
        String dataset = "AllGestureWiimoteZ";
        String filepath = "D:\\Research TSC\\Data\\Univariate2018_ts\\" + dataset + "\\" + dataset;
        String filepath_orig = "D:\\Research TSC\\Data\\TSCProblems2018\\" + dataset + "\\" + dataset;
        //String test = "ItalyPowerDemand_TEST.ts";

        //File f = new File("D:\\Research TSC\\Data\\test.ts");
        File f = new File(filepath + "_TRAIN" + ".ts");
        System.out.println(f);
        TSReader ts_reader = new TSReader(new FileReader(f));
        Instances train_data = ts_reader.GetInstances();


        File f_orig = new File(filepath_orig);
        Instances train_data_orig = new Instances(new FileReader(f_orig));

        //do some comparison!

        System.out.println(train_data.toString());
    }
}
