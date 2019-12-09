package utilities;

import weka.core.*;
import weka.core.converters.AbstractFileLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.Properties;

public class TSReader {

    public static final String PROBLEM_NAME = "@problemName";
    public static final String TIME_STAMPS = "@timeStamps";
    public static final String CLASS_LABEL = "@classLabel";
    public static final String UNIVARIATE = "@univariate";
    public static final String DATA = "@data";


    private final StreamTokenizer m_Tokenizer;
    private int m_Lines;

    private boolean multivariate = false;

    Instances m_data;
    private String problemName;
    private boolean timeStamps;
    private boolean univariate;
    private boolean classLabel;
    private ArrayList<String> classLabels;

    public TSReader(Reader reader) throws IOException{
        m_Tokenizer = new StreamTokenizer(reader);
        initTokenizer();

        readHeader(1000);
        //initBuffers

        //read each line and extract a data Instance
        Instance inst;
        while((inst = readInstance(m_data)) != null){
            m_data.add(inst);
        }
    }

    public Instances GetInstances(){
        return m_data;
    }

    private Instance readInstance(Instances m_data) throws IOException{
        if(multivariate){
            return readMultivariateInstance(m_data);
        }
        else{
            return readUnivariateInstance(m_data);
        }
    }

    private Instance readMultivariateInstance(Instances m_data) throws IOException {
        return null;
    }

    private Instance readUnivariateInstance(Instances m_data) throws IOException {

        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            return null;
        }


        ArrayList<Double> timeSeries = new ArrayList<>();
        String classValue = null;

        //read the tokens, and if we hit a : then we need to do something clever.
        boolean bFoundColon = false;
        do{
            //this means we're either about to get the class value or we need to check whether we're a multivariate series.
            if(m_Tokenizer.ttype == ':'){
                bFoundColon = true;
            }
            else{
                if(bFoundColon){
                    System.out.println("Class Value: "+ m_Tokenizer.sval);
                    classValue = m_Tokenizer.sval;
                }
                else{
                    timeSeries.add(m_Tokenizer.nval);
                    System.out.println(m_Tokenizer.sval);
                }
            }

            m_Tokenizer.nextToken();
        } while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);


        //this is the first time series read.
        //which means we need to build the attribute data // header file stuff.
        //then create an Instance from that. etc.
        if(this.m_data == null){
            // create attribute list
            ArrayList<Attribute> attList = new ArrayList<>();
            for(int i = 0; i < timeSeries.size(); i++){
                attList.add(new Attribute("att"+i));
            }
            //have to cast the null to arraylist type (this is taken from WEKAS OWN CODE.) to force it to use the right method.
            attList.add(new Attribute("classVal", (ArrayList<String>)null, classLabels.size()));


            this.m_data = new Instances(problemName, attList, attList.size());
        }


        Instance inst = new DenseInstance(timeSeries.size()+1);
        for(int a = 0; a < timeSeries.size(); a++){
            inst.setValue(a, timeSeries.get(a));
        }
        inst.setValue(timeSeries.size()-1, classLabels.indexOf(classValue));

        return inst;
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
    protected void readHeader(int capacity) throws IOException {
        m_Lines = 0;
        String relationName = "";

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
            System.out.println(m_Tokenizer.sval);

            getNextToken();
            //now read all the class values until we reach the EOL
            do{
                classLabels.add(m_Tokenizer.sval);

                System.out.println(m_Tokenizer.sval);

                m_Tokenizer.nextToken();
            }while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);
        } else {
            errorMessage("keyword " + PROBLEM_NAME + " expected");
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
            System.out.println(m_Tokenizer.sval);
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
        String filepath = "D:\\Research TSC\\Data\\Univariate2018_ts\\ItalyPowerDemand\\";
        String train = "ItalyPowerDemand_TRAIN.ts";
        String test = "ItalyPowerDemand_TEST.ts";

        TSReader ts_reader = new TSReader(new FileReader(new File(filepath+train)));
        Instances train_data = ts_reader.GetInstances();

        System.out.println(train_data.toString());
    }
}
