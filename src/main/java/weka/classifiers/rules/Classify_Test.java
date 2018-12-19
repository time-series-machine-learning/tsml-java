package weka.classifiers.rules;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Serializable;
import java.util.Hashtable;
import java.util.StringTokenizer;

import weka.core.Instance;

public class Classify_Test  implements Serializable {

    private static final long serialVersionUID = 1L;


    public static void main(String[] args) {
    }

    // majority selection of the class . Returns -1 if classification fails, else returns position of the class label in the vector
    static int maggioranza(String[] rules,String[] class_labels, String class_path, String pathname, Instance instance) {
        //read class_labels and id_class_base
        int id_base_class = 0;
        int class_labels_freq[] = new int[class_labels.length];
        for (int i = 0; i< class_labels.length; i++)
            class_labels_freq[i] = 0; // frequency counter initially set to zero
        FileReader file = null;
        StringTokenizer st;
        try {
            file = new FileReader(class_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean eof = false;
        boolean firstline = true;
        String line = null;
        try {   // Legge da file le etichette di classe
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    if (firstline) {
                        id_base_class = Integer.parseInt(line);
                        firstline = false;
                    }
                }
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // scan rules counting class_labels appearance
        int counter;
        for (int i = 0; i < rules.length; i++) {
            if (rules[i] != null && rules[i]!="") {
                StringTokenizer st2 = new StringTokenizer(rules[i], " ");
                counter = 0;
                while (st2.hasMoreTokens()) {
                    String s = st2.nextToken();
                    counter++;
                    if (counter == 3) {
                        // string s cointains class label
                        /*** non weighted majority selection ***/
                        //Integer.parseInt(s)-id_base_class is the position of the class in the class_labels vector
                        class_labels_freq[Integer.parseInt(s)-id_base_class]++;

                    }
                } // end while
            } // end if
        }

        // search for the maximum
        int maximum = -1;
        int result = -1;
        for (int i= 0; i<class_labels_freq.length; i++) {
            if (class_labels_freq[i]>maximum) {
                result = i;
                maximum = class_labels_freq[i];
            }
        }

        instance.setClassValue(class_labels[result]);
        int my_result = (int) instance.value(instance.classIndex());
        return (my_result);
    }


    static // majority selection of the class . Returns -1 if classification fails, else returns position of the class label in the vector
    int maggioranza_numeric(String[] rules,String[] class_labels, String class_path, String pathname) {
        //read class_labels and id_class_base
        int id_base_class = 0;
        int class_labels_freq[] = new int[class_labels.length];
        for (int i = 0; i< class_labels.length; i++)
            class_labels_freq[i] = 0; // frequency counter initially set to zero
        FileReader file = null;
        StringTokenizer st;
        try {
            file = new FileReader(class_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean eof = false;
        boolean firstline = true;
        String line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    if (firstline) {
                        id_base_class = Integer.parseInt(line);
                        firstline = false;
                    }
                }
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // scan rules counting class_labels appearance
        int counter;
        for (int i = 0; i < rules.length; i++) {
            if (rules[i] != null && rules[i]!="") {
                StringTokenizer st2 = new StringTokenizer(rules[i], " ");
                counter = 0;
                while (st2.hasMoreTokens()) {
                    String s = st2.nextToken();
                    counter++;
                    if (counter == 3) {
                        // string s cointains class label
                        /*** non weighted majority selection ***/
                        class_labels_freq[Integer.parseInt(s)-id_base_class]++;
                    }
                } // end while
            } // end if
        }

        // search for the maximum
        int maximum = -1;
        int result = -1;
        for (int i= 0; i<class_labels_freq.length; i++) {
            if (class_labels_freq[i]>maximum) {
                result = i;
                maximum = class_labels_freq[i];
            }
        }
        return (Integer.parseInt(class_labels[result]));
    }

    static int elimina(String[] rules, double soglia) {
        int i;
        double conf;
        double rule_conf = 0.0;
        StringTokenizer st;
        String s;
        int counter = 0;
        // read rhe rule confidence
        st = new StringTokenizer(rules[0], " ");
        while (st.hasMoreTokens()) {
            s = st.nextToken();
            counter++;
            if (counter == 5) // confidence value
                rule_conf = Double.parseDouble(s);
        }
        conf = rule_conf - soglia;
        for (i = 1; rules[i] != null; i++) {
            // read confidence of i-th rule
            st = new StringTokenizer(rules[i], " ");
            while (st.hasMoreTokens()) {
                s = st.nextToken();
                counter++;
                if (counter == 5) // confidence value
                    rule_conf = Double.parseDouble(s);
            } // end while
            if ( rule_conf <= conf ) {
                /* Elimino la regola e le successive */
                for (int j = i; j< rules.length; j++)
                    rules[j] = null;
                return (i);
            }
        } // end for
        return (i); 
    }


    double[] search_row_NEW(String current_bin_path, String dataset, Instance instance, String[] class_labels, int id_class_base) {
        // read the original arff dataset searching for the correpondent row in the binary file
        int row = -1;
        String my_String = "";
        double[] instance_vector = new double[instance.numAttributes()-1];
        for (int i= 0; i< instance.numAttributes()-1; i++) {
            //if (i != (instance.numAttributes()-1))
            my_String += Integer.toString((int)instance.value(i));
            if (i != (instance.numAttributes()-1))
                my_String += ",";
        }
        FileReader file = null;
        try {
            file = new FileReader(dataset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean start = false;
        boolean eof = false;
        String line = null;
        int row_counter = 0;
        try {
            while (!eof) {
                line = buff.readLine();
                if (start == true)
                    row_counter++;
                if (line == null)
                    eof = true;
                else {
                    if (start) {
                        line = line.substring(0, line.lastIndexOf(",")+1);
                    }
                    if (line.equalsIgnoreCase(my_String) == true) {
                        row = (row_counter-1);
                    }
                    if (line.equalsIgnoreCase("@data")==true)
                        start = true;
                }
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (row == -1)  {
            return null;
        }

        //read the corresponent row-th row in the binary file
        //update the instance and return it.
        RandomAccessFile filebin = null;
        BinaryFile binFile;

        // set the endian mode to LITTLE_ENDIAN
        final short endian = BinaryFile.LITTLE_ENDIAN;
        // set the signed mode to unsigned
        final boolean signed = false;
        long tid, cid, numItems, item;
        long i;
        try {
            filebin = new RandomAccessFile(current_bin_path, "r");
            binFile = new BinaryFile(filebin);
            // set the endian mode to LITTLE_ENDIAN
            binFile.setEndian(BinaryFile.LITTLE_ENDIAN);
            // set the signed mode to unsigned
            binFile.setSigned(false);
            while (true) {
                // read tid, cid, and number of items
                tid=binFile.readDWord();
                cid=binFile.readDWord();
                numItems=binFile.readDWord();

                for (i=0;i<numItems-1;i++)
                {
                    item=binFile.readDWord();
                    if (tid == row)
                        instance_vector[(int)i]= (double)item;
                }
                item=binFile.readDWord();
            }
        } catch (Exception e) {
            System.out.println("**Error: " + e.getMessage());
        }
        try {
            filebin.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return instance_vector;
    }

    static //	 selection of the first num_rules rules that classify the transaction
    String[] selection_NEW(double[] transaction, String levelI_path, String levelII_path, int num_rules, int num_features) {
        String[] rules = new String[num_rules];
        int index = 0;
        for (int i = 0; i < num_rules; i++)
            rules[i] = null;
        // scan levelI rules and verify if all the items in each rule are contained into the transaction
        FileReader file = null;
        StringTokenizer st;
        try {
            file = new FileReader(levelI_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean eof = false;
        boolean found = false;
        String line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    st = new StringTokenizer(line, ",");
                    while (st.hasMoreTokens()) {
                        found = false;
                        String s = st.nextToken();
                        if (s.indexOf("{")!=-1 && s.indexOf("}")==-1)
                            s = s.substring(1,s.length());
                        if (s.indexOf("{")!=-1 && s.indexOf("}")!=-1)
                            s = s.substring(1,s.length()-1);
                        if (s.indexOf("{")==-1 && s.indexOf("}")!=-1)
                            s = s.substring(0,s.indexOf("}"));
                        //System.out.println("s: "+s+" ");
                        for (int j = 0; (j< num_features) && (found == false); j++) {
                            if (transaction[j] == Double.parseDouble(s)) found = true;
                        }
                        if (!found)
                            break; // element not found.. pass to the next rule
                    }
                    if (found) { // if finishing inspecting a rule found remains true, it means that a matching rule
                        rules[index]= new String(line);
                        index++;
                        if (index == (num_rules))
                            return rules;
                    } // end if
                } // end else
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        /*** possible modification: if I find one level-I rule exit  ***/
        if (rules != null)
            return rules;
        /*** end modification ***/

// if program arrives here, it means that not enough rules could be extracted from levelI.. let's move to levelII
        file = null;
        st = null;
        try {
            file = new FileReader(levelII_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        buff = new BufferedReader(file);
        eof = false;
        found = false;
        line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    st = new StringTokenizer(line, ",");
                    while (st.hasMoreTokens()) {
                        found = false;
                        String s = st.nextToken();
                        if (s.indexOf("{")!=-1 && s.indexOf("}")==-1)
                            s = s.substring(1,s.length());
                        if (s.indexOf("{")!=-1 && s.indexOf("}")!=-1)
                            s = s.substring(1,s.length()-1);
                        if (s.indexOf("{")==-1 && s.indexOf("}")!=-1)
                            s = s.substring(0,s.indexOf("}"));
                        for (int j = 0; (j< num_features) && (found == false); j++) {
                            if (transaction[j] == Double.parseDouble(s)) found = true;
                        }
                        if (!found)
                            break; // element not found.. pass to the next rule
                    }
                    if (found) { // if finishing inspecting a rule found remains true, it means that a matching rule
                        rules[index]= new String(line);
                        index++;
                        if (index == (num_rules))
                            return rules;
                    } // end if
                } // end else
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

// if program arrives here, it means that not enough rules could be extracted from levelI or II..
// return Null for remaining not extracted rules
        return rules;
    }


    static //	 selection of the first num_rules rules that classify the transaction
    String[] selection_HASH(String[] transaction, String levelI_path, String levelII_path, int num_rules, int num_features, Hashtable hash) {
        if (hash.isEmpty()== true) {
            //System.err.println("Error on hash table!\n");
            return null;
        }
        String[] rules = new String[num_rules];
        int index = 0;
        for (int i = 0; i < num_rules; i++)
            rules[i] = null;
        // scan levelI rules and verify if all the items in each rule are contained into the transaction
        FileReader file = null;
        StringTokenizer st;
        try {
            file = new FileReader(levelI_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean eof = false;
        boolean found = false;
        String line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    st = new StringTokenizer(line, ",");
                    while (st.hasMoreTokens()) {
                        found = false;
                        String s = st.nextToken();
                        if (s.indexOf("{")!=-1 && s.indexOf("}")==-1)
                            s = s.substring(1,s.length());
                        else {
                            if (s.indexOf("{")!=-1 && s.indexOf("}")!=-1)
                                s = s.substring(1,s.indexOf("}"));
                            else {
                                if (s.indexOf("{")==-1 && s.indexOf("}")!=-1)
                                    s = s.substring(0,s.indexOf("}"));
                            }
                        }
                        for (int j = 0; (j< (transaction.length)) && (found == false); j++) {
                            String myStr = "Attr"+j+"Value"+transaction[j];
                            Integer n = (Integer)hash.get(myStr);
                            if (n != null) {
                                if (Integer.parseInt(s) == n.intValue())
                                    found = true;
                            } else {
                                // if the element doesn't appear in the hashtable you cannot say anything about it
                                //found = true;
                            }
                        }
                        if (!found)
                            break; // element not found.. pass to the next rule
                    }
                    if (found) { // if finishing inspecting a rule found remains true, it means that a matching rule
                        rules[index]= new String(line);
                        index++;
                        if (index == (num_rules))
                            return rules;
                    } // end if
                } // end else
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        /*** possible modification: if I find one level-I rule exit  ***/
        if (index > 0)
            return rules;
        /*** end modification ***/

// if program arrives here, it means that not enough rules could be extracted from levelI.. let's move to levelII
//System.out.println("Pass to level II..");
        file = null;
        st = null;
        try {
            file = new FileReader(levelII_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        buff = new BufferedReader(file);
        eof = false;
        found = false;
        line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    st = new StringTokenizer(line, ",");
                    while (st.hasMoreTokens()) {
                        found = false;
                        String s = st.nextToken();
                        if (s.indexOf("{")!=-1 && s.indexOf("}")==-1)
                            s = s.substring(1,s.length());
                        else {
                            if (s.indexOf("{")!=-1 && s.indexOf("}")!=-1)
                                s = s.substring(1,s.indexOf("}"));
                            else {
                                if (s.indexOf("{")==-1 && s.indexOf("}")!=-1)
                                    s = s.substring(0,s.indexOf("}"));
                            }
                        }
                        //System.out.println("s: "+s+" ");
                        for (int j = 0; (j< transaction.length) && (found == false); j++) {
                            String myStr = "Attr"+j+"Value"+transaction[j];
                            Integer n = (Integer)hash.get(myStr);
                            if (n != null) {
                                if (Integer.parseInt(s) == n.intValue())
                                    found = true;
                            } else {
                                // if the element doesn't appear in the hashtable you cannot say anything about it
                                //found = true;
                            }
                        }
                        if (!found)
                            break; // element not found.. pass to the next rule
                    }
                    if (found) { // if finishing inspecting a rule found remains true, it means that a matching rule
                        rules[index]= new String(line);
                        index++;
                        if (index == (num_rules))
                            return rules;
                    } // end if
                } // end else
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

// if program arrives here, it means that not enough rules could be extracted from levelI or II..
// return Null for remaining not extracted rules
        return rules;
    }


    static //	 selection of the first num_rules rules that classify the transaction
    String[] selection_HASH_numeric(double[] transaction, String levelI_path, String levelII_path, int num_rules, int num_features, Hashtable hash) {
        if (hash.isEmpty()== true) {
            System.err.println("Error on hash table!\n");
            return null;
        }
        String[] rules = new String[num_rules];
        int index = 0;
        //double[] transaction = new double[num_features+1];
        // read the list of transactions
        //transaction = Test_instance.toDoubleArray();
        for (int i = 0; i < num_rules; i++)
            rules[i] = null;
        // scan levelI rules and verify if all the items in each rule are contained into the transaction
        FileReader file = null;
        StringTokenizer st;
        try {
            file = new FileReader(levelI_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean eof = false;
        boolean found = false;
        String line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    st = new StringTokenizer(line, ",");
                    while (st.hasMoreTokens()) {
                        found = false;
                        String s = st.nextToken();
                        if (s.indexOf("{")!=-1 && s.indexOf("}")==-1)
                            s = s.substring(1,s.length());
                        else {
                            if (s.indexOf("{")!=-1 && s.indexOf("}")!=-1)
                                s = s.substring(1,s.indexOf("}"));
                            else {
                                if (s.indexOf("{")==-1 && s.indexOf("}")!=-1)
                                    s = s.substring(0,s.indexOf("}"));
                            }
                        }
                        //System.out.println("s: "+s+" ");
                        for (int j = 0; (j< transaction.length) && (found == false); j++) {
                            //if (transaction[j] == Double.parseDouble(s)) found = true;
                            //	 found = true;
                            String myStr = "Attr"+j+"Value"+(int)transaction[j];
                            Integer n = (Integer)hash.get(myStr);
                            if (n != null) {
                                if (Double.parseDouble(s) == n.intValue())
                                    found = true;
                            } else {
                                // if the element doesn't appear in the hashtable you cannot say anything about it
                                //found = true;
                            }
                        }
                        if (!found)
                            break; // element not found.. pass to the next rule
                    }
                    if (found) { // if finishing inspecting a rule found remains true, it means that a matching rule
                        rules[index]= new String(line);
                        index++;
                        if (index == (num_rules))
                            return rules;
                    } // end if
                } // end else
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        /*** possible modification: if I find one level-I rule exit  ***/
        if (index > 0)
            return rules;
        /*** end modification ***/

// if program arrives here, it means that not enough rules could be extracted from levelI.. let's move to levelII
//System.out.println("pass to level II..");
        file = null;
        st = null;
        try {
            file = new FileReader(levelII_path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        buff = new BufferedReader(file);
        eof = false;
        found = false;
        line = null;
        try {
            while (!eof) {
                line = buff.readLine();
                if (line == null)
                    eof = true;
                else {
                    st = new StringTokenizer(line, ",");
                    while (st.hasMoreTokens()) {
                        found = false;
                        String s = st.nextToken();
                        if (s.indexOf("{")!=-1 && s.indexOf("}")==-1)
                            s = s.substring(1,s.length());
                        else {
                            if (s.indexOf("{")!=-1 && s.indexOf("}")!=-1)
                                s = s.substring(1,s.indexOf("}"));
                            else {
                                if (s.indexOf("{")==-1 && s.indexOf("}")!=-1)
                                    s = s.substring(0,s.indexOf("}"));
                            }
                        }
                        for (int j = 0; (j< num_features) && (found == false); j++) {
                            //if (transaction[j] == Double.parseDouble(s)) found = true;
                            //	 found = true;
                            String myStr = "Attr"+j+"Value"+Double.parseDouble(s);
                            Integer n = (Integer)hash.get(myStr);
                            if (n != null) {
                                if (transaction[j] == n.intValue())
                                    found = true;
                            } else {
                                // if the element doesn't appear in the hashtable you cannot say anything about it

                            }
                        }
                        if (!found)
                            break; // element not found.. pass to the next rule
                    }
                    if (found) { // if finishing inspecting a rule found remains true, it means that a matching rule
                        rules[index]= new String(line);
                        index++;
                        if (index == (num_rules))
                            return rules;
                    } // end if
                } // end else
            }	// end while (!eof)
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

// if program arrives here, it means that not enough rules could be extracted from levelI or II..
// return Null for remaining not extracted rules
        return rules;
    }


} // end class
