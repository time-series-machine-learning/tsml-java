package experiments.data;

import utilities.generic_storage.Pair;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

import static utilities.multivariate_tools.MultivariateInstanceTools.createRelationHeader;

/**
File for reading sktime format data

 @author Aaron Bostrom, pushed 22/4/2020
 */

public class TSReader {


    //need to change this to a map function.
    public static final String PROBLEM_NAME = "@problemName";
    public static final String TIME_STAMPS = "@timeStamps";
    public static final String CLASS_LABEL = "@classLabel";
    public static final String UNIVARIATE = "@univariate";
    public static final String MISSING = "@missing";
    public static final String DATA = "@data";


    private HashMap<String, String> variables = new HashMap<>();


    private final StreamTokenizer m_Tokenizer;
    private int m_Lines;

    Instances m_data;
    private String problemName;
    private boolean univariate = false;
    private boolean missing = false;
    private boolean timeStamps = false;
    private boolean classLabel;
    private ArrayList<String> classLabels;
    private ArrayList<Attribute> attList;

    private ArrayList<ArrayList<Double>> uni_raw_data;
    private ArrayList<ArrayList<ArrayList<Double>>> multi_raw_data;

    private ArrayList<Double> raw_labels;

    public TSReader(Reader reader) throws IOException{
        m_Tokenizer = new StreamTokenizer(reader);
        initTokenizer();

        readHeader();

        if(univariate){
            CreateUnivariateInstances();
        }
        else{
            CreateMultivariateInstances();
        }

    }

    private void CreateMultivariateInstances() throws IOException {
        multi_raw_data = new ArrayList<>();
        raw_labels = new ArrayList<>();

        //read each line and extract a data Instance
        Pair<ArrayList<ArrayList<Double>>, Double> multi_series_and_label;
        //extract the multivariate series, and the possible label.
        while(( multi_series_and_label = readMultivariateInstance()) != null){
            multi_raw_data.add(multi_series_and_label.var1);
            raw_labels.add(multi_series_and_label.var2);
        }

        //go through all the raw data, and find the longest row.
        int max_length = 0;
        for(ArrayList<ArrayList<Double>> channel : multi_raw_data){
            int curr = channel.stream().mapToInt(ArrayList::size).max().getAsInt();
            if(curr > max_length)
                max_length = curr;
        }


        int numAttsInChannel=max_length;
        int numChannels = multi_raw_data.get(0).size(); //each array in this list is a channel.


        // create attribute list
        attList = new ArrayList<>();

        //construct relational attribute.#
        Instances relationHeader = createRelationHeader(numAttsInChannel,numChannels);
        relationHeader.setRelationName("relationalAtt");
        Attribute relational_att = new Attribute("relationalAtt", relationHeader, numAttsInChannel);
        attList.add(relational_att);

        if(classLabel)
            attList.add(new Attribute("classVal", classLabels, classLabels.size()));

        m_data = new Instances(problemName, attList, multi_raw_data.size());
        for(int i=0; i< multi_raw_data.size(); i++){

            ArrayList<ArrayList<Double>> series = multi_raw_data.get(i);
            m_data.add(new DenseInstance(attList.size()));

            //TODO: add all the time series values, dealing with missing values.
            Instances relational = new Instances(relationHeader, series.size());
    
            //each dense instance is row/ which is actually a channel.
            for(int k=0; k< series.size(); k++){
                
                DenseInstance ds = new DenseInstance(numAttsInChannel);
                int index  = 0;
                for(Double d : series.get(k))
                    ds.setValue(index++, d);

                relational.add(ds);
            }            

            //add the relational series to the attribute, and set the value of the att to the relations index.
            int index = m_data.instance(i).attribute(0).addRelation(relational);
            //System.out.println(index);
            m_data.instance(i).setValue(0, index);

            //set class value.
            if(classLabel) {
                m_data.instance(i).setValue(1, raw_labels.get(i));
            }
        }



    }

    private void CreateUnivariateInstances() throws IOException {
        uni_raw_data = new ArrayList<>();
        raw_labels = new ArrayList<>();
        //read each line and extract a data Instance
        Pair<ArrayList<Double>, Double>  series_and_label;
        //extract series and the possible label.
        while((series_and_label = readUnivariateInstance()) != null){
            uni_raw_data.add(series_and_label.var1);
            raw_labels.add(series_and_label.var2);
        }

        //go through all the raw data, and find the longest row.
        int max_length = uni_raw_data.stream().mapToInt(ArrayList::size).max().getAsInt();

        // create attribute list
        attList = new ArrayList<>();
        for(int i = 0; i < max_length; i++){
            attList.add(new Attribute("att"+(i+1), i));
        }
        //have to cast the null to arraylist type (this is taken from WEKAS OWN CODE.) to force it to use the right method.

        if(classLabel)
            attList.add(new Attribute("classVal", classLabels, classLabels.size()));

        this.m_data = new Instances(problemName, attList, uni_raw_data.size());


        for(int i=0; i<uni_raw_data.size(); i++){
            ArrayList<Double> timeSeries = uni_raw_data.get(i);
            //add all the time series values.
            Instance ds = new DenseInstance(max_length+1);
                int index  = 0;
                for(Double d : timeSeries)
                    ds.setValue(index++, d);
            //only add if we have a classLabel
            //get the value from the end of the current time series, and put it at the end of the attribute list.
            if(classLabel) {
                ds.setValue(max_length, raw_labels.get(i));
            }
            this.m_data.add(ds);
        }
    }

    public Instances GetInstances(){
        return m_data;
    }

    private Pair<ArrayList<ArrayList<Double>>, Double> readMultivariateInstance() throws IOException {
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            return null;
        }

        ArrayList<ArrayList<Double>> multi_timeSeries = new ArrayList<>();
        String classValue ="";


        ArrayList<Double> timeSeries = new ArrayList<>();
        do{
            //this means we're about to get the class value
            if(m_Tokenizer.ttype == ':' && classLabel){
                //add the current time series to the list.
                multi_timeSeries.add(timeSeries);
                timeSeries = new ArrayList<>();
            }
            else{               
                timeSeries.add(m_Tokenizer.sval == "?" ? Double.NaN : m_Tokenizer.nval);
                classValue = m_Tokenizer.sval == null ? ""+m_Tokenizer.nval : m_Tokenizer.sval; //the last value to be tokenized should be the class value. can be in string or number format so check both.
            }
            m_Tokenizer.nextToken();
        } while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);
        
        //don't add the last series to the list, instead extract the first element and figure out what the class value is.
        double classVal = classLabel ? (double) this.classLabels.indexOf(classValue) : -1.0;
        return new Pair<>(multi_timeSeries,classVal);
    }

    private  Pair<ArrayList<Double>, Double> readUnivariateInstance() throws IOException {
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
                    classValue = m_Tokenizer.sval == null ? ""+m_Tokenizer.nval : m_Tokenizer.sval;
                    bFoundColon = false;
                }
                else{
                    //if the tokenizer has a ? in it, then we set the value to nan.
                    timeSeries.add(m_Tokenizer.sval == "?" ? Double.NaN : m_Tokenizer.nval);
                }
            }
            m_Tokenizer.nextToken();
        } while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);

        double classVal = classLabel ? (double)this.classLabels.indexOf(classValue) : -1.0;
        return new Pair<>(timeSeries, classVal);
    }

    private void initTokenizer() {
        //Setup the tokenizer to read the stream.
        m_Tokenizer.resetSyntax();

        ////ignore 0-9 chars
        m_Tokenizer.wordChars(' '+1,'\u00FF');

        m_Tokenizer.parseNumbers();

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

        //this gets the token there may be weirdness at the front of the file.
        getFirstToken();
        if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
            errorMessage("premature end of file");
        }

        do{
            
            String token = m_Tokenizer.sval;

            if(token.equalsIgnoreCase(CLASS_LABEL)){
                ExtractClassLabels();
            }
            else{
                variables.put(token, ExtractVariable(token));
            }


            getNextToken();

        }while(!m_Tokenizer.sval.equalsIgnoreCase(DATA));


        //these are required.
        problemName = variables.get(PROBLEM_NAME);
        if (problemName == null){
            errorMessage("keyword " + PROBLEM_NAME + " expected");
        }

        if (variables.get(UNIVARIATE) == null){
            errorMessage("keyword " + UNIVARIATE + " expected");
        }
        else{
            univariate = Boolean.parseBoolean(variables.get(UNIVARIATE));
        }

        //set optionals.
        if(variables.get(MISSING) != null)
            missing = Boolean.parseBoolean(variables.get(MISSING));
        if(variables.get(TIME_STAMPS) != null)
            timeStamps = Boolean.parseBoolean(variables.get(TIME_STAMPS));



        //clear out last tokens.
        getLastToken(false);
    }

    private void ExtractClassLabels() throws IOException {
        classLabels = new ArrayList<>();
        getNextToken();
        classLabel = Boolean.parseBoolean(m_Tokenizer.sval);

        if(!classLabel) {
            getLastToken(false);
            return;
        }

        getNextToken();
        //now read all the class values until we reach the EOL
        do{
            classLabels.add(m_Tokenizer.sval == null ? ""+m_Tokenizer.nval : m_Tokenizer.sval);
            m_Tokenizer.nextToken();
        }while(m_Tokenizer.ttype != StreamTokenizer.TT_EOL);
    }

    private String ExtractVariable(String VARIABLE) throws IOException {
        //check if the current token matches the hardcoded value for @types e.g. @problemName etc.
        getNextToken();
        String value = m_Tokenizer.sval;
        getLastToken(false);
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

        String local_path = "D:\\Work\\Data\\Univariate_ts\\";
        String local_path_orig = "D:\\Work\\Data\\Univariate_arff\\";
        String m_local_path = "D:\\Work\\Data\\Multivariate_ts\\";
        String m_local_path_orig = "D:\\Work\\Data\\Multivariate_arff\\";


        //for(String dataset : DatasetLists.tscProblems2018){
            String dataset = "AllGestureWiimoteZ";
            String filepath = local_path + dataset + "\\" + dataset;
            String filepath_orig = local_path_orig + dataset + "\\" + dataset;

            File f = new File(filepath + "_TRAIN" + ".ts");
            //System.out.println(f);

            long time = System.nanoTime();
            TSReader ts_reader = new TSReader(new FileReader(f));
            System.out.println("after: " + (System.nanoTime() - time));

            Instances train_data = ts_reader.GetInstances();
            //System.out.println(train_data);
        //}

        //File f_orig = new File(filepath_orig);
        //Instances train_data_orig = new Instances(new FileReader(f_orig));

        //System.out.println(train_data.toString());

        //for(String dataset_multi :  DatasetLists.mtscProblems2018){
            String dataset_multi = "CharacterTrajectories";
            String filepath_multi = m_local_path + dataset_multi + "\\" + dataset_multi;
            String filepath_orig_multi = m_local_path_orig + dataset_multi + "\\" + dataset_multi;

            File f1 = new File(filepath_multi + "_TRAIN" + ".ts");
            System.out.println(f1);
            time = System.nanoTime();
            TSReader ts_reader_multi = new TSReader(new FileReader(f1));
            Instances train_data_multi = ts_reader_multi.GetInstances();
            System.out.println("after: " + (System.nanoTime() - time));


            //JAMESL ADDED TESTS
            //Instances tsisntances = DatasetLoading.loadData(filepath_multi + "_TRAIN");
        //}

        //File f_orig_multi = new File(filepath_orig_multi);
        //Instances train_data_orig_multi = new Instances(new FileReader(f_orig_multi));

        //System.out.println(train_data_multi.instance(0));

        //do some comparison!
        //System.out.println(train_data_multi.toString());

    }
}
