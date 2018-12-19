package weka.classifiers.rules;

import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.io.IOException;



public class Lazy_Pruning implements Serializable {

    public static void lazy_main(String[] args) {
        //inputs
        String Rules_File = new String();
        String Transactions_File = new String();
        String Level1_Rules_File = new String();
        String Level2_Rules_File = new String();

        //checking for correct input
        if (args.length != 4) {
            System.out.println("Wrong number of parameters, you must enter four file addresses, in the following order:");
            System.out.println("file with mined rules");
            System.out.println("file with transactions");
            System.out.println("text file where level1 rules should be stored");
            System.out.println("text file where level2 rules should be stored");
            System.exit(1);
        } else {
            Rules_File = args[0];
            Transactions_File = args[1];
            Level1_Rules_File = args[2];
            Level2_Rules_File = args[3];
        }

        //initialising the rule and tranaction lists
        ArrayList<RuleL3> List_of_Rules = new ArrayList<RuleL3>();
        ArrayList<Transaction> List_of_Transactions = new ArrayList<Transaction>();

        //extracting rules from the file and populating the list of rules
        List_of_Rules = readRules(Rules_File);

        //populating the list of transactions
        List_of_Transactions = populateTransactions(Transactions_File);

        //running the algorithm
        LazyPruning(List_of_Rules,List_of_Transactions,Level1_Rules_File,Level2_Rules_File);

    }

    //this method extracts rules from the file and returns the list of rules
    public static ArrayList<RuleL3> readRules(String rules_file) {
        ArrayList<RuleL3> list_of_rules = new ArrayList<RuleL3>();
        String next_rule;
        int rule_no = 0;
        try {
            BufferedReader in = new BufferedReader(new FileReader(rules_file));
            if (!in.ready())
                throw new IOException();

            while ((next_rule = in.readLine()) != null) {
                list_of_rules.add(new RuleL3(next_rule,rule_no++));
            }
            in.close();
        } catch (IOException e) {
            // File vuoto o inesistente. Vado avanti comunque e genero
            // un modello vuoto.
            // Il file vuoto può essere dato dalla mancanza di attributi
            // (solo quello di classe e basta)
            //System.out.println(e);
        }
        return list_of_rules;
    }

    //this method reads integers from a binary file and returns a them as an ArrayList
    public static ArrayList<Integer> readData (String filename) {
        File file = null;
        boolean file_end = false;
        int    i_data = 0;
        String little_endian;
        String[] big_endian = new String[4];
        StringBuilder s_builder;
        ArrayList<Integer> integers = new ArrayList<Integer>();

        file = new File (filename);

        try {
            FileInputStream file_input = new FileInputStream (file);
            DataInputStream data_in    = new DataInputStream (file_input );

            while (!file_end) {
                try {
                    i_data = data_in.readInt ();
                    Integer i_b = new Integer(i_data);
                    //System.err.println(i_b.intValue());
                    little_endian = Integer.toBinaryString(i_data);
                    s_builder = new StringBuilder();
                    if (little_endian.length()!=32) {
                        for (int j=0;j<32-little_endian.length();j++)
                            s_builder.append(0);
                        little_endian = new String(s_builder)+little_endian;
                    }

                    for (int k = 1; k <= 4; k++) {
                        big_endian[k-1] = little_endian.substring(8*(k-1),8*k);
                    }

                    integers.add(Integer.parseInt(big_endian[3]+big_endian[2]+big_endian[1]+big_endian[0],2));

                } catch (EOFException eof) {
                    file_end = true;
                }
            }
            data_in.close ();
        } catch  (IOException e) {
            System.out.println ( "IO Exception =: " + e );
        }
        return integers;
    }

    //this method gets the name of the file containing transactions and returns a list of transactions
    public static ArrayList<Transaction> populateTransactions(String transactions_file) {
        int counter=0;
        ArrayList<Transaction> transaction_list = new ArrayList<Transaction>();
        int tid=0;
        int cid=0;
        int Class_ID;
        int Num_Items;
        Item Items[];
        int length;

        ArrayList<Integer> integers = readData(transactions_file);

        while (counter!=integers.size()) {
            tid = integers.get(counter++).intValue();
            //System.err.println("tid: "+tid);
            cid = integers.get(counter++).intValue();
            //System.err.println("cid: "+cid);
            length = integers.get(counter++).intValue();
            //System.err.println("length: "+length);
            Num_Items = length-1;
            Items = new Item[Num_Items];
            for (int j = 0; j < Num_Items; j++) {
                Items[j] = new Item(integers.get(counter++).intValue());
            }
            Class_ID = integers.get(counter++).intValue();
            transaction_list.add(new Transaction(tid,cid,Class_ID,Num_Items,Items));
        }

        return transaction_list;
    }

    //this is the core method that implements the lazypruning algorithm and writes rules into level1.txt and level2.txt
    public static void LazyPruning(ArrayList<RuleL3> list_of_rules,ArrayList<Transaction> list_of_transactions,String level1_rules_file,String level2_rules_file ) {
        int rule_number = 0;
        int transaction_number = 0;
        try {
            FileWriter outFile1 = new FileWriter(level1_rules_file, false);
            PrintWriter out1 = new PrintWriter(outFile1);
            FileWriter outFile2 = new FileWriter(level2_rules_file, false);
            PrintWriter out2 = new PrintWriter(outFile2);

            while (rule_number < list_of_rules.size()) {

                transaction_number = 0;
                while (transaction_number < list_of_transactions.size()) {
                    list_of_rules.get(rule_number).classifyTrans(list_of_transactions.get(transaction_number));
                    transaction_number++;

                }

                list_of_rules.get(rule_number).setLevel();

                if (list_of_rules.get(rule_number).getCorrect() > 0) {
                    transaction_number = 0;
                    while (transaction_number < list_of_transactions.size()) {
                        if (list_of_transactions.get(transaction_number).getLast_Rule() == list_of_rules.get(rule_number).getRule_ID()) {
                            list_of_transactions.remove(transaction_number);
                            transaction_number = 0;
                        } else
                            transaction_number++;
                    }
                }

                String rule= "{";
                for (int i = 0; i < list_of_rules.get(rule_number).getLength(); i++) {
                    rule = rule+Integer.toString((list_of_rules.get(rule_number).getItems()[i].getValue()));
                    if (i!=list_of_rules.get(rule_number).getLength()-1)
                        rule = rule+",";
                }
                rule = rule+"} -> "+Integer.toString(list_of_rules.get(rule_number).getClass_ID())
                       +" "+Integer.toString((list_of_rules.get(rule_number).getAbsolute_Support()))
                       +" "+Float.toString((list_of_rules.get(rule_number).getConfidence()))
                       +" "+Integer.toString((list_of_rules.get(rule_number).getCorrect()))
                       +" "+Integer.toString((list_of_rules.get(rule_number).getIncorrect()));

                if (list_of_rules.get(rule_number).getLevel()==1)
                    out1.println(rule);
                else if (list_of_rules.get(rule_number).getLevel()==2)
                    out2.println(rule);


                rule_number++;
            }
            out1.close();
            out2.close();

        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    public void print_Transactions_List(ArrayList<Transaction> list_of_transactions) {
        int transaction_number = 0;
        while (transaction_number < list_of_transactions.size()) {
            System.out.println(list_of_transactions.get(transaction_number).gettid());
            System.out.println(list_of_transactions.get(transaction_number).getcid());
            //System.out.println(list_of_transactions.get(transaction_number).getItems());
            System.out.println(list_of_transactions.get(transaction_number).getNum_Items());
            System.out.println(list_of_transactions.get(transaction_number).getClass_ID());
            transaction_number++;
        }
    }

}



