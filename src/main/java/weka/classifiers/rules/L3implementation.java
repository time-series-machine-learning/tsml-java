package weka.classifiers.rules;

import java.lang.*;
import java.io.*;
import java.util.*;

import weka.core.Instances;
public class L3implementation implements Serializable {

    int CLASS_NUM = 200;
    String[] class_labels;
    public int num_features;
    public int num_samples;
    public int num_classes;

    public void get_header(String dataset_name, String format) {
        try {
            if (format.equalsIgnoreCase("arff")== true) { // let's suppose arff format
                // count the number of rows starting with "@attribute" and subtract
                // the last one refering to the class labels.
                FileReader file = new FileReader(dataset_name);
                BufferedReader buff = new BufferedReader(file);
                boolean eof = false;
                boolean start_count_samples = false;
                String line = null;
                String s = null;
                String s2 = null;
                StringTokenizer st, st2;
                int count_genes = 0; // number of "@attribute" string occurences
                int count_classes = 0; // number of class labels occurences in the last attribute definition
                int count_samples = 0; // number of samples equal to the number of lines after "@data" line
                // read the input file
                while (!eof) {
                    line = buff.readLine();
                    // System.out.println("la riga letta �: "+line2);
                    if (line == null)
                        eof = true;
                    else {
                        st = new StringTokenizer(line);
                        while (st.hasMoreTokens()) {
                            s = st.nextToken();
                            if (s.equalsIgnoreCase("@attribute"))
                                count_genes++;
                            if (s.indexOf("{")!= -1) {
                                // s contains the list of possibile class labels
                                st2 = new StringTokenizer(s, ",");
                                while (st2.hasMoreTokens()) {
                                    s2 = st2.nextToken();
                                    //class_labels[count_classes]= new String(s2);
                                    count_classes++;
                                }
                            }
                            if (start_count_samples == true)
                                count_samples++;
                            if (s.equalsIgnoreCase("@data")== true)
                                start_count_samples = true;
                        }

                    } // end else	if (line2 == null)
                }	// end while (!eof)
                num_features = count_genes-1;
                //num_classes = count_classes;
                num_samples = count_samples;
            } // end else: arff format
            else { 
                // let's suppose csv format
                FileReader file = new FileReader(dataset_name);
                BufferedReader buff = new BufferedReader(file);
                boolean eof = false;
                boolean start_count_samples = false;
                String line = null;
                String s = null;
                String s2 = null;
                boolean firstline = true;
                StringTokenizer st, st2;
                int count_features = 0;
                int count_samples = 0;
                int count_classes = 0;
                // read the input file
                while (!eof) {
                    line = buff.readLine();
                    // System.out.println("la riga letta �: "+line2);
                    if (line == null)
                        eof = true;
                    else {
                        if (firstline) {
                            st = new StringTokenizer(line, ",");
                            while (st.hasMoreTokens()) {
                                s = st.nextToken();
                                count_features++;
                            }
                            count_features--; // ignore class attribute
                        } else {
                            count_samples++;
                        }
                    } // end else
                }	// end while (!eof)
                num_features = count_features;
                //num_classes = count_classes;
                num_samples = count_samples;
            }
        } catch (IOException e) {
            e.getMessage();
        }
    }

    public String[] read_Class_Labels_not_well_formed(String dataset_name, String format) {
        String[] temp = new String[100];
        FileReader file;
        boolean eof = false;
        boolean start = false;
        String line = null;
        String line2 = null;
        StringTokenizer st;
        int count_features = 0;
        num_classes = 0;
        try {
            file = new FileReader(dataset_name);
            BufferedReader buff = new BufferedReader(file);
            if (format.equalsIgnoreCase("Arff")==true) {
                while (!eof) {
                    line = buff.readLine();
                    if (line == null)
                        eof = true;
                    else {
                        if (start == true) {
                            st = new StringTokenizer(line, ",");
                            while (st.hasMoreTokens()) {
                                String s = st.nextToken();
                                if (count_features == num_features) {
                                    // check if the class label already exists
                                    boolean found = false;
                                    for (int i = 0; i<num_classes; i++)
                                        if (temp[i].equalsIgnoreCase(s)== true) found = true;
                                    if (!found) {
                                        num_classes++;
                                        temp[num_classes-1] = new String(s);
                                    }
                                }
                                count_features++;
                            }
                        }
                        if ((line.equalsIgnoreCase("@data")== true))
                            start = true; //from the next cycle let's start reading!
                    }
                    count_features = 0;
                }
            } // end if (format == arff)

            if (format.equalsIgnoreCase("csv")==true) {
                // read the first header line counting the number of features
                line = buff.readLine();
                if (line == null)
                    return null; // error!!
                num_features = -1;
                st = new StringTokenizer(line, ",");
                while (st.hasMoreTokens()) {
                    String s2 = st.nextToken();
                    num_features++;
                }
                while (!eof) {
                    line = buff.readLine();
                    if (line == null)
                        eof = true;
                    else {
                        st = new StringTokenizer(line, ",");
                        while (st.hasMoreTokens()) {
                            String s = st.nextToken();
                            if (count_features == num_features) {
                                // check if the class label already exists
                                boolean found = false;
                                for (int i = 0; i<num_classes; i++)
                                    if (temp[i].equalsIgnoreCase(s)== true) found = true;
                                if (!found) {
                                    num_classes++;
                                    temp[num_classes-1] = new String(s);
                                }
                            }
                            count_features++;
                        }
                    }
                    count_features = 0;
                }
            } // end if (format == csv)
            if (format.equalsIgnoreCase("data")==true) {
                // read the first header line counting the number of features
                line = buff.readLine();
                if (line == null)
                    return null; // error!!
                num_features = -1;
                st = new StringTokenizer(line, ",");
                while (st.hasMoreTokens()) {
                    String s2 = st.nextToken();
                    num_features++;
                }
                // restart reading the file
                file.close();
                buff.close();
                file = new FileReader(dataset_name);
                buff = new BufferedReader(file);
                while (!eof) {
                    line = buff.readLine();
                    if (line == null)
                        eof = true;
                    else {
                        st = new StringTokenizer(line, ",");
                        while (st.hasMoreTokens()) {
                            String s = st.nextToken();
                            if (count_features == num_features) {
                                // check if the class label already exists
                                boolean found = false;
                                for (int i = 0; i<num_classes; i++)
                                    if (temp[i].equalsIgnoreCase(s)== true) found = true;
                                if (!found) {
                                    num_classes++;
                                    temp[num_classes-1] = new String(s);
                                }
                            }
                            count_features++;
                        }
                    }
                    count_features = 0;
                }
            } // end if (format == data)
        } catch (IOException e) {
            e.printStackTrace();
        }
        // copy temp values in class_labels vector
        class_labels = new String[num_classes];
        for (int i = 0; i< num_classes; i++)
            class_labels[i] = new String(temp[i]);
        return class_labels;
    }

    String[] read_Class_Labels_well_formed(String dataset_name, String format) {
        class_labels = new String[num_classes];
        try {
            if (format.equalsIgnoreCase("arff")== true) { // let's suppose arff format
                // count the number of rows starting with "@attribute" and subtract
                // the last one refering to the class labels.
                FileReader file = new FileReader(dataset_name);
                BufferedReader buff = new BufferedReader(file);
                boolean eof = false;
                boolean start_count_samples = false;
                String line = null;
                String s = null;
                String s2 = null;
                StringTokenizer st, st2;
                int count_genes = 0; // number of "@attribute" string occurences
                int count_classes = 0; // number of class labels occurences in the last attribute definition
                int count_samples = 0; // number of samples equal to the number of lines after "@data" line
                // read the input file

                while (!eof) {
                    line = buff.readLine();
                    if (line == null)
                        eof = true;
                    else {
                        st = new StringTokenizer(line);
                        while (st.hasMoreTokens()) {
                            s = st.nextToken();
                            if (s.equalsIgnoreCase("@attribute"))
                                count_genes++;
                            if (s.indexOf("{")!= -1) {
                                // s contains the list of possibile class labels
                                st2 = new StringTokenizer(s, ",");
                                while (st2.hasMoreTokens()) {
                                    s2 = st2.nextToken();
                                    if (s2.indexOf("{")!= -1)
                                        s2 = s2.substring(s2.indexOf("{")+1, s2.length());
                                    if (s2.indexOf("}")!= -1)
                                        s2 = s2.substring(0, s2.indexOf("}"));
                                    class_labels[count_classes]= new String(s2);
                                    count_classes++;
                                }
                            }
                        }

                    } // end else	if (line2 == null)
                }	// end while (!eof)
            } // end else: arff format
        } catch (IOException e) {
            e.getMessage();
        }
        return class_labels;
    }


    public Hashtable creabin(String filename, int label, String[] class_labels, Hashtable hash, String class_path, String bin_name) {
        // read transactions file and save them into a binary file
        RandomAccessFile file;
        BinaryFile binFile;
        // set the endian mode to LITTLE_ENDIAN
        final short endian = BinaryFile.LITTLE_ENDIAN;
        // set the signed mode to unsigned
        final boolean signed = false;
        boolean eof = false;
        int j = 0;
        BufferedReader buff_rd;
        FileReader file_rd;
        FileWriter cl_file = null;
        boolean start = false; // start to read after "@data" line
        int attrib_counter= 0;
        String line2 = null;
        StringTokenizer st;
        Integer n2;
        String myStr2 = null;
        int current_value = 0;
        long t_id = 0;
        long c_id = 0;


        try {
            file_rd = new FileReader(filename);
            buff_rd = new BufferedReader(file_rd);

            file = new RandomAccessFile(bin_name, "rw");
            binFile = new BinaryFile(file);
            // set the endian mode to LITTLE_ENDIAN
            try {
                binFile.setEndian(BinaryFile.LITTLE_ENDIAN);
            } catch (Exception e) {
                e.printStackTrace();
            }
            // set the signed mode to unsigned
            binFile.setSigned(false);

            if (label == 1) {
                // create also class labels file
                cl_file = new FileWriter(class_path);
                cl_file.write(Integer.toString(CLASS_NUM)+"\n");

                for (int i = 0; i< num_classes; i++) {
                    cl_file.write(class_labels[i]+"\n");
                }
                cl_file.close();
            }

            while (!eof) {
                line2 = buff_rd.readLine();
                if (line2 == null)
                    eof = true;
                else {
                    if (start == true) {
                        st = new StringTokenizer(line2, ",");

                        binFile.writeDWord(t_id);
                        binFile.writeDWord(c_id);
                        t_id++;
                        c_id++;
                        // write X number of objects to be read in the current rows, class label included
                        //file_wr.write(num_features);
                        binFile.writeDWord((long)(num_features+1));
                        while (st.hasMoreTokens()) {
                            String s = st.nextToken();
                            if ((attrib_counter == num_features)  && (s!= null)) {   
                                // last column contains class label for that gene
                                // write the correspondent number of the class
                                for (j= 0; j< num_classes; j++) // look for the index of the class label
                                    if (class_labels[j].equals(s) == true) break;
                                binFile.writeDWord((long)(CLASS_NUM+j));
                            } // end if
                            else {   // Attributo predittivo e non di classe
                                if (s!= null) {
                                    //here I have to check if the couple (attr, value) already exists.
                                    //if yes return the correspondent number and write down it in the binary file
                                    // otherwise add the couple to the hash table writing down in the binary file
                                    // the correspondent number of the first avaible couple associated to that attribute
                                    String myStr = "Attr"+attrib_counter+"Value"+s;
                                    Integer n = (Integer)hash.get(myStr);
                                    if (n != null) {
                                        // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                        binFile.writeDWord(n.longValue());
                                    } else {
                                        // value assigned is the next incremental value available
                                        current_value++;
                                        hash.put(myStr, new Integer(current_value));
                                        binFile.writeDWord((long)current_value);
                                    }
                                }
                            }
                            attrib_counter++;
                        }
                    } // end if (start== true)
                    if ((line2.equalsIgnoreCase("@data")== true))
                        start = true; //from the next cycle let's start reading!
                } // end else (line == null)
                attrib_counter = 0;
            }
            buff_rd.close();
            file_rd.close();
            binFile = null;
            file.close();
        } catch (IOException e) {
            e.getMessage();
        }

        return hash;
    }

    public void leggiDWordbin(String filename) {
        RandomAccessFile file;
        BinaryFile binFile;

// set the endian mode to LITTLE_ENDIAN
        final short endian = BinaryFile.LITTLE_ENDIAN;
// set the signed mode to unsigned
        final boolean signed = false;

        long tid, cid, numItems, item;
        long i;

        try {
            file = new RandomAccessFile(filename, "r");
            binFile = new BinaryFile(file);

            // set the endian mode to LITTLE_ENDIAN
            binFile.setEndian(BinaryFile.LITTLE_ENDIAN);
            // set the signed mode to unsigned
            binFile.setSigned(false);


            while (true) {
                // read tid, cid, and number of items
                tid=binFile.readDWord();
                cid=binFile.readDWord();
                numItems=binFile.readDWord();

                for (i=0;i<numItems-1;i++) {
                    item=binFile.readDWord();
                }

                item=binFile.readDWord();
            }


        } catch (Exception e) {
            System.out.println("**Error: " + e.getMessage());
        }
    }

    public void leggiWordbin(String filename) {
        RandomAccessFile file;
        BinaryFile binFile;

//	 set the endian mode to LITTLE_ENDIAN
        final short endian = BinaryFile.LITTLE_ENDIAN;
//	 set the signed mode to unsigned
        final boolean signed = false;

        int tid, cid, numItems, item;
        long i;

        try {
            //System.out.println("Open file "+filename);
            file = new RandomAccessFile(filename, "r");
            binFile = new BinaryFile(file);

            // set the endian mode to LITTLE_ENDIAN
            binFile.setEndian(BinaryFile.LITTLE_ENDIAN);
            // set the signed mode to unsigned
            binFile.setSigned(false);

            while (true) {
                // read tid, cid, and number of items
                tid=binFile.readWord();
                cid=binFile.readWord();
                numItems=binFile.readWord();

                for (i=0;i<numItems-1;i++) {
                    item=binFile.readWord();
                }

                item=binFile.readWord();
            }


        } catch (Exception e) {
            System.out.println("**Error: " + e.getMessage());
        }
    }




    public void readbin(String filename) {
        try {
            FileInputStream file = new FileInputStream(filename);
            boolean eof = false;
            int count = 0;
            while (!eof) {
                int input = file.read();
                //System.out.println("input"+input+"\n");
                if (input == -1)
                    eof = true;
                else
                    count++;
            }
            file.close();
            //System.out.println("Byte read: "+count+"\n");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    String create_arff_from_data(String dataset, String path_name) {
        String arff_filename = path_name + "dataset_converted.Arff";
        try {
            //count the number of attributes
            FileReader file_rd = new FileReader(dataset);
            BufferedReader buff_rd = new BufferedReader(file_rd);
            // read the first line and parse different elements in order to understand number and types of attributes
            // let's suppose that class attribute is last one by default
            String line2 = buff_rd.readLine();
            StringTokenizer st = new StringTokenizer(line2, ",");
            // count the number of attributes
            int attributes = 0;
            String s;
            while (st.hasMoreTokens()) {
                s = st.nextToken();
                attributes++;
            }
            FileWriter fw = new FileWriter(arff_filename);
            // write the header part
            fw.write("@relation Arff_dataset"+"\n"+"\n");
            for (int counter = 1; counter < (attributes+1); counter++) {
                if (counter != attributes)
                    fw.write("@attribute "+"attr"+counter+" numeric"+"\n");
                else {
                    fw.write("@attribute class {");
                    for (int i=0; i< class_labels.length; i++) {
                        if (i!= class_labels.length-1)
                            fw.write(class_labels[i]+",");
                        else
                            fw.write(class_labels[i]+"}"+"\n");
                    }
                }
            }
            fw.write("\n"+"@data");
            buff_rd.close();
            file_rd.close();
            boolean eof = false;
            file_rd = new FileReader(dataset);
            buff_rd = new BufferedReader(file_rd);
            while (!eof) {
                line2 = buff_rd.readLine();
                if (line2 == null)
                    eof = true;
                else {
                    //write the line in the arff data file
                    fw.write("\n");
                    fw.write(line2);
                }
            }
            fw.close();
            buff_rd.close();
            file_rd.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arff_filename;
    }

    String create_arff_from_csv(String dataset, String pathname) {
        String arff_filename = null;
        String conversion_path = null;
        try {
            //count the number of attributes
            FileReader file_rd = new FileReader(dataset);
            BufferedReader buff_rd = new BufferedReader(file_rd);
            // read the first line and parse different elements in order to understand number and types of attributes
            // let's suppose that class attribute is last one by default
            String line2 = buff_rd.readLine();
            StringTokenizer st = new StringTokenizer(line2, ",");
            // count the number of attributes
            int attributes = 0;
            String s;
            while (st.hasMoreTokens()) {
                s = st.nextToken();
                attributes++;
            }
            arff_filename = dataset.substring(dataset.lastIndexOf("/")+1,dataset.lastIndexOf("."));
            conversion_path =pathname+ arff_filename+".Arff";
            FileWriter fw = new FileWriter(conversion_path);
            // write the header part
            fw.write("@relation Arff_dataset"+"\n"+"\n");
            for (int counter = 1; counter < (attributes+1); counter++) {
                if (counter != attributes)
                    fw.write("@attribute "+counter+" string"+"\n");
                else {
                    fw.write("@attribute class string"+"\n");
                }
            }
            fw.write("@data");
            boolean eof = false;
            while (!eof) {
                line2 = buff_rd.readLine();
                if (line2 == null)
                    eof = true;
                else {
                    fw.write("\n");
                    fw.write(line2);
                }
            }
            fw.close();
            buff_rd.close();
            file_rd.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return conversion_path;
    }



    void converti_in_numerico(String dataset, String numerico_path, String format) {

        boolean eof = false;
        int j = 0;
        BufferedReader buff_rd;
        FileReader file_rd;
        FileWriter cl_file = null;
        boolean start = false; // start to read after "@data" line
        int attrib_counter= 0;
        int sample_index=0;
        int feature = -1;
        String line2 = null;
        StringTokenizer st;
        Integer n2;
        String myStr2 = null;
        int current_value = 0;
        int current_class_value = 0;
        Hashtable hash = new Hashtable();

        try {
            file_rd = new FileReader(dataset);
            buff_rd = new BufferedReader(file_rd);
            FileWriter fw = new FileWriter(numerico_path);
            if (format.equalsIgnoreCase("Arff")== true) {
                while (!eof) {
                    line2 = buff_rd.readLine();
                    if (line2 == null)
                        eof = true;
                    else {
                        if (start == true) {
                            st = new StringTokenizer(line2, ",");
                            while (st.hasMoreTokens()) {
                                String s = st.nextToken();
                                if ((attrib_counter == (feature))  && (s!= null)) {
                                    // last column contains class label for that gene
                                    // here I have to check if the couple (attr, value) already exists.
                                    // if yes return the correspondent number and write down it in the binary file
                                    // otherwise add the couple to the hash table writing down in the binary file
                                    // the correspondent number of the first avaible couple associated to that attribute
                                    String myStr = "Attr"+attrib_counter+"Value"+s;
                                    Integer n = (Integer)hash.get(myStr);
                                    if (n != null) {
                                        // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                        //file_wr.write(n.intValue());
                                        fw.write(n+"\n");
                                    } else {
                                        // value assigned is the next incremental value aviable
                                        current_class_value++;
                                        hash.put(myStr, new Integer(current_class_value));
                                        fw.write(current_class_value+"\n");
                                    }
                                } // end if
                                else {
                                    if (s!= null) {
                                        //here I have to check if the couple (attr, value) already exists.
                                        //if yes return the correspondent number and write down it in the binary file
                                        // otherwise add the couple to the hash table writing down in the binary file
                                        // the correspondent number of the first avaible couple associated to that attribute
                                        String myStr = "Attr"+attrib_counter+"Value"+s;
                                        Integer n = (Integer)hash.get(myStr);
                                        if (n != null) {
                                            // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                            //file_wr.write(n.intValue());
                                            fw.write(n+",");
                                        } else {
                                            // value assigned is the next incremental value aviable
                                            current_value++;
                                            hash.put(myStr, new Integer(current_value));
                                            fw.write(current_value+",");
                                        }
                                    }
                                }
                                attrib_counter++;
                            }
                        } // end if (start== true)
                        else {
                            fw.write(line2+"\n");
                            if (line2.contains("@attribute")== true)
                                feature++;
                        }
                        if ((line2.equalsIgnoreCase("@data")== true))
                            start = true; //from the next cycle let's start reading!
                    } // end else (line == null)
                    attrib_counter = 0;
                }
            }

            if (format.equalsIgnoreCase("csv")== true) {
                while (!eof) {
                    line2 = buff_rd.readLine();
                    if (line2 == null)
                        eof = true;
                    else {
                        if (start == true) {
                            st = new StringTokenizer(line2, ",");
                            while (st.hasMoreTokens()) {
                                String s = st.nextToken();
                                if ((attrib_counter == (feature))  && (s!= null)) {
                                    // last column contains class label for that gene
                                    // here I have to check if the couple (attr, value) already exists.
                                    // if yes return the correspondent number and write down it in the binary file
                                    // otherwise add the couple to the hash table writing down in the binary file
                                    // the correspondent number of the first avaible couple associated to that attribute
                                    String myStr = "Attr"+attrib_counter+"Value"+s;
                                    Integer n = (Integer)hash.get(myStr);
                                    if (n != null) {
                                        // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                        fw.write(n+"\n");
                                    } else {
                                        // value assigned is the next incremental value aviable
                                        current_class_value++;
                                        hash.put(myStr, new Integer(current_class_value));
                                        fw.write(current_class_value+"\n");
                                    }
                                } // end if
                                else {
                                    if (s!= null) {
                                        //here I have to check if the couple (attr, value) already exists.
                                        //if yes return the correspondent number and write down it in the binary file
                                        // otherwise add the couple to the hash table writing down in the binary file
                                        // the correspondent number of the first avaible couple associated to that attribute
                                        String myStr = "Attr"+attrib_counter+"Value"+s;
                                        Integer n = (Integer)hash.get(myStr);
                                        if (n != null) {
                                            // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                            fw.write(n+",");
                                        } else {
                                            // value assigned is the next incremental value aviable
                                            current_value++;
                                            hash.put(myStr, new Integer(current_value));
                                            fw.write(current_value+",");
                                        }
                                    }
                                }
                                attrib_counter++;
                            }
                        } // end if (start== true)
                        else {
                            fw.write(line2+"\n");
                            StringTokenizer st2;
                            st2 = new StringTokenizer(line2, ",");
                            while (st2.hasMoreTokens()) {
                                String s2 = st2.nextToken();
                                feature++;
                            }
                            start = true;
                        }
                    } // end else (line == null)
                    attrib_counter = 0;
                }
            }
            if (format.equalsIgnoreCase("data")== true) {
                boolean firstline = true;
                while (!eof) {
                    line2 = buff_rd.readLine();
                    if (firstline) {
                        StringTokenizer st3;
                        st3 = new StringTokenizer(line2, ",");
                        while (st3.hasMoreTokens()) {
                            String s3 = st3.nextToken();
                            feature++;
                        }
                        firstline = false;
                    }
                    if (line2 == null)
                        eof = true;
                    else {
                        st = new StringTokenizer(line2, ",");
                        while (st.hasMoreTokens()) {
                            String s = st.nextToken();
                            if ((attrib_counter == (feature))  && (s!= null)) {
                                // last column contains class label for that gene
                                // here I have to check if the couple (attr, value) already exists.
                                // if yes return the correspondent number and write down it in the binary file
                                // otherwise add the couple to the hash table writing down in the binary file
                                // the correspondent number of the first avaible couple associated to that attribute
                                String myStr = "Attr"+attrib_counter+"Value"+s;
                                Integer n = (Integer)hash.get(myStr);
                                if (n != null) {
                                    // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                    //file_wr.write(n.intValue());
                                    fw.write(n+"\n");
                                } else {
                                    // value assigned is the next incremental value aviable
                                    hash.put(myStr, new Integer(current_class_value));
                                    //file_wr.write((int)current_value);
                                    fw.write(current_class_value+"\n");
                                    current_class_value++;
                                }
                            } // end if
                            else {
                                if (s!= null) {
                                    //here I have to check if the couple (attr, value) already exists.
                                    //if yes return the correspondent number and write down it in the binary file
                                    // otherwise add the couple to the hash table writing down in the binary file
                                    // the correspondent number of the first avaible couple associated to that attribute
                                    String myStr = "Attr"+attrib_counter+"Value"+s;
                                    Integer n = (Integer)hash.get(myStr);
                                    if (n != null) {
                                        // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                                        //file_wr.write(n.intValue());
                                        fw.write(n+",");
                                    } else {
                                        // value assigned is the next incremental value aviable
                                        current_value++;
                                        hash.put(myStr, new Integer(current_value));
                                        //file_wr.write((int)current_value);
                                        fw.write(current_value+",");
                                    }
                                }
                            }
                            attrib_counter++;
                        }
                    } // end else (line == null)
                    attrib_counter = 0;
                }
            }
            buff_rd.close();
            fw.close();
            file_rd.close();
        } catch (IOException e) {
            e.getMessage();
        }
    }

    public String[] read_Class_Labels_from_instances(Instances inst) {
        num_classes = inst.numClasses();
        String[] classes = new String[num_classes];
        //initialization
        for (int g = 0; g < num_classes; g++)
            classes[g] = "";
        int cur_num_classes = -1;
        boolean found = false;
        for (int i = 0; i < inst.numInstances(); i++) {
            if (cur_num_classes >= 0) // search for the current class label
                for (int k = 0; k < (cur_num_classes) && (found== false) ; k++) {
                    if (inst.instance(i).classValue() == Double.parseDouble(classes[k]))
                        found = true;
                }
            if (!found) {
                if (cur_num_classes == -1)
                    cur_num_classes++;
                classes[cur_num_classes]= new String(String.valueOf(inst.instance(i).classValue()));
                cur_num_classes++;
            }
            found = false;
        }
        return classes;
    }

    public Hashtable creabin_from_instances(Instances inst, int label, String[] class_labels, Hashtable hash, String pathname, String dataset, int current_fold) {
        // read transactions file and save them into a binary file
        RandomAccessFile file;
        BinaryFile binFile;
        // set the endian mode to LITTLE_ENDIAN
        final short endian = BinaryFile.LITTLE_ENDIAN;
        // set the signed mode to unsigned
        final boolean signed = false;
        int j = 0;
        FileWriter cl_file = null;
        int current_value = 0;
        long t_id = 0;
        long c_id = 0;
        boolean found = false;


        try {
            String dataset_name = dataset.substring(dataset.lastIndexOf("/")+1, dataset.lastIndexOf("."));
            String bin_name = pathname + dataset_name + "k"+ current_fold +".bin";
            String class_path = pathname + dataset_name + "k"+ current_fold + ".cls";
            //System.out.println("class path name: "+ class_path+"\n");
            //System.out.println("bin_name: "+bin_name+"\n");
            file = new RandomAccessFile(bin_name, "rw");
            binFile = new BinaryFile(file);
            // set the endian mode to LITTLE_ENDIAN
            try {
                binFile.setEndian(BinaryFile.LITTLE_ENDIAN);
            } catch (Exception e) {
                e.printStackTrace();
            }
            // set the signed mode to unsigned
            binFile.setSigned(false);
            num_classes = inst.numClasses();
            if (label == 1) {
                // create also class labels file
                cl_file = new FileWriter(class_path);
                cl_file.write(Integer.toString(CLASS_NUM)+"\n");
                for (int i = 0; i< num_classes; i++) {
                    cl_file.write(class_labels[i]+"\n");
                }
                cl_file.close();
            }
            for (int i = 0; i < inst.numInstances(); i++) {
                binFile.writeDWord(t_id);
                binFile.writeDWord(c_id);
                t_id++;
                c_id++;
                // write X number of objects to be read in the current rows, class label included
                //file_wr.write(num_features);
                binFile.writeDWord((long)(inst.firstInstance().numAttributes()));
                for (int k = 0; k < inst.instance(i).numAttributes()-1; k++) {
                    // hash table handling
                    // here I have to check if the couple (attr, value) already exists.
                    // if yes return the correspondent number and write down it in the binary file
                    // otherwise add the couple to the hash table writing down in the binary file
                    // the correspondent number of the first avaible couple associated to that attribute
                    String myStr = "Attr"+k+"Value"+String.valueOf(inst.instance(i).value(k));
                    Integer n = (Integer)hash.get(myStr);
                    if (n != null) {
                        // I find the key (Attrib, Value) so we must write the correspondent number n into the binary file
                        binFile.writeDWord(n.longValue());
                    } else {
                        // value assigned is the next incremental value aviable
                        current_value++;
                        hash.put(myStr, new Integer(current_value));
                        binFile.writeDWord((long)current_value);
                    }
                }
                for (j= 0; j< num_classes; j++) // look for the index of the class label
                    if (Double.parseDouble(class_labels[j])==(inst.instance(i).classValue()) == true)
                        break;
                binFile.writeDWord((long)(CLASS_NUM+j));
                found = false;
            }
            file.close();
        } catch (IOException e) {
            e.getMessage();
        }
        return hash;
    }


    String create_csv_from_data(String dataset, String[] class_labels, String path_name) {
        String csv_filename = path_name + "dataset_converted.csv";
        try {
            //count the number of attributes
            FileReader file_rd = new FileReader(dataset);
            BufferedReader buff_rd = new BufferedReader(file_rd);
            // read the first line and parse different elements in order to understand number and types of attributes
            // let's suppose that class attribute is last one by default
            String line2 = buff_rd.readLine();
            StringTokenizer st = new StringTokenizer(line2, ",");
            // count the number of attributes
            int attributes = 0;
            String s;
            while (st.hasMoreTokens()) {
                s = st.nextToken();
                attributes++;
            }
            FileWriter fw = new FileWriter(csv_filename);
            // write the header part
            for (int counter = 1; counter < (attributes+1); counter++) {
                if (counter != attributes)
                    fw.write("Attr"+counter+",");
                else {
                    fw.write("class");
                }
            }
            buff_rd.close();
            file_rd.close();
            boolean eof = false;
            file_rd = new FileReader(dataset);
            buff_rd = new BufferedReader(file_rd);
            while (!eof) {
                line2 = buff_rd.readLine();
                if (line2 == null)
                    eof = true;
                else {
                    //write the line in the arff data file
                    fw.write("\n");
                    fw.write(line2);
                }
            }
            fw.close();
            buff_rd.close();
            file_rd.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return csv_filename;
    }

    void update_class_values(String training_filename, String training_filename_new, String[] class_labels) {

        try {
            FileReader file_rd = new FileReader(training_filename);
            BufferedReader buff_rd = new BufferedReader(file_rd);
            FileWriter fw = new FileWriter(training_filename_new);
            // write the same file except for the attribute class
            boolean eof = false;
            String line = null;
            int contatore = 1;
            while (!eof) {
                line = buff_rd.readLine();
                if (line == null)
                    eof = true;
                else {
                    //write the line in the arff data file
                    if (line.indexOf("@attribute")==-1)
                        fw.write(line+"\n");
                    else {
                        if (line.indexOf("@attribute class")!=-1) {
                            // write the full class list
                            fw.write("@attribute class {");
                            for (int i= 0; i < class_labels.length; i++) {
                                fw.write("'"+class_labels[i]+"'");
                                if (i !=(class_labels.length -1) )
                                    fw.write(",");
                            }
                            fw.write("}");
                            fw.write("\n");
                        } else {
                            String linea = "@attribute attr"+contatore+" String";
                            fw.write(linea+"\n");
                            contatore++;
                        }

                    }
                }
            }
            fw.close();
            buff_rd.close();
            file_rd.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
