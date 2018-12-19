package weka.classifiers.rules.ruleshandler;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class CMain implements Serializable {

    static public int idBaseClasse;
    static public int idMassimoClasse;
    static public int[] suppClasses;
    static public double transazioniTotali;
    static public int MAX_CLASSES = 30; // Da modificare anche nella classe CClasse
    static public double conf_threshold;
    static public double supp_threshold;

    public static void my_main(String[] args) {


        if (args.length !=5 ) {
            System.out.println("Usage: fp_new_comb <TDB> <support threshold%%> <confidence threshold%%> <stem output file> <pahtname>\n");
            System.exit(1);
        }


        supp_threshold = Double.parseDouble(args[1]);
        conf_threshold = Double.parseDouble(args[2]);

        suppClasses = new int[MAX_CLASSES];

        for ( int i = 0 ; i<MAX_CLASSES ; i++) {
            suppClasses[i] = 0;
        }



        itemClasse(args[0]);

        supportCountingClasses(args[0]);

        CClasse[] cc = new CClasse[(idMassimoClasse-idBaseClasse)+1];

        for ( int i = 0 ; i<(idMassimoClasse-idBaseClasse)+1 ; i++) {
            cc[i] = new CClasse();
            cc[i].estraiPerClasse(args.length,args,i);
        }


        return;

    }
    public static int itemClasse(String fileName) {

        int classe = 0;
        int minimo = 0;
        int massimo = 0;
        int flag = 0;
        byte b;
        int n = 0;

        try {


            FileInputStream fis = new FileInputStream ( fileName );

            DataInputStream di = new DataInputStream(fis);



            for ( int h = 0 ; h<3 ; h++ ) {

                ByteBuffer bf = ByteBuffer.allocate(4);
                for ( int k = 0 ; k<4 ; k++ ) {
                    b = di.readByte();

                    bf.order(ByteOrder.LITTLE_ENDIAN);

                    bf.put(b);
                }
                n = bf.getInt(0);

            }

            for ( int f = 0 ; f<n ; f++) {
                ByteBuffer buf = ByteBuffer.allocate(4);
                for ( int g = 0 ; g<4 ; g++ ) {
                    b = di.readByte();

                    buf.order(ByteOrder.LITTLE_ENDIAN);

                    buf.put(b);

                }
                classe = buf.getInt(0);

            }
            minimo = massimo = classe;

            flag++;

            while ( true ) {
                for ( int h = 0 ; h<3 ; h++ ) {

                    ByteBuffer bf = ByteBuffer.allocate(4);
                    for ( int k = 0 ; k<4 ; k++ ) {
                        b = di.readByte();

                        bf.order(ByteOrder.LITTLE_ENDIAN);

                        bf.put(b);
                    }
                    n = bf.getInt(0);

                }

                for ( int f = 0 ; f<n ; f++) {
                    ByteBuffer buf = ByteBuffer.allocate(4);
                    for ( int g = 0 ; g<4 ; g++ ) {
                        b = di.readByte();

                        buf.order(ByteOrder.LITTLE_ENDIAN);

                        buf.put(b);

                    }
                    classe = buf.getInt(0);

                }

                if ( classe < minimo) {
                    minimo = classe;

                }
                if ( classe > massimo) {
                    massimo = classe;

                }

            }

        } catch (EOFException eofx) {

            if ( flag == 0 ) {
                System.err.println("Numero di transazioni pari a 0");
                System.exit(1);
            }


            idBaseClasse = minimo;
            idMassimoClasse = massimo;



            return(minimo);

        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return 0;

    }

    public static void supportCountingClasses(String fileName) {

        byte b;
        int n = 0;
        int classe = 0;

        try {


            FileInputStream fis = new FileInputStream ( fileName );

            DataInputStream di = new DataInputStream(fis);


            while ( true ) {
                for ( int h = 0 ; h<3 ; h++ ) {

                    ByteBuffer bf = ByteBuffer.allocate(4);
                    for ( int k = 0 ; k<4 ; k++ ) {
                        b = di.readByte();

                        bf.order(ByteOrder.LITTLE_ENDIAN);

                        bf.put(b);
                    }
                    n = bf.getInt(0);

                }

                for ( int f = 0 ; f<n ; f++) {
                    ByteBuffer buf = ByteBuffer.allocate(4);
                    for ( int g = 0 ; g<4 ; g++ ) {
                        b = di.readByte();

                        buf.order(ByteOrder.LITTLE_ENDIAN);

                        buf.put(b);

                    }
                    classe = buf.getInt(0);

                }
                suppClasses[classe-idBaseClasse]++;
                transazioniTotali++;

            }


        } catch (EOFException eofx) {

            return ;

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return;
    }

    public int count_num_of_rules(String filename) {
        int num = 0;
        FileReader file = null;
        try {
            file = new FileReader(filename);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader buff = new BufferedReader(file);
        boolean eof = false;
        String line = null;
        while (!eof) {
            try {
                line = buff.readLine();
                num++;
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (line == null)
                eof = true;
        }	// end while (!eof)
        try {
            file.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return num;
    }

    public void mergeRuleFiles(String rules_path, int classes, int current_fold, String path_name) {
        FileReader file = null;
        FileWriter fw = null;
        String filename = null;
        try {
            fw = new FileWriter(rules_path);
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error in opening rules during merge..");

        }
        for (int i = 0; i< classes; i++) {
            filename = path_name+"c"+ i+"rules-k"+String.valueOf(current_fold)+ ".txt";
            File my_file = new File(filename);
            if (my_file.exists()) {
                try {
                    file = new FileReader(filename);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Error in opening rules during merge..");
                }
                BufferedReader buff = new BufferedReader(file);
                boolean eof = false;
                String line = null;
                try {
                    while (!eof) {
                        line = buff.readLine();
                        if (line == null)
                            eof = true;
                        else
                            fw.write(line+"\n");
                    }	// end while (!eof)
                    file.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }// end if (my_file.exists())
        } // end for
        try {
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

