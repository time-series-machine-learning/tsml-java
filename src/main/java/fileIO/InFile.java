package fileIO;
import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
public class InFile{
    private String fileName;
    private FileReader fr;
    private BufferedReader in;
    private StreamTokenizer token;
    private StreamTokenizer markerToken;
    private static int MAXBUFFERSIZE =1000000;
//Standard File
    public InFile(String name){
        try{
            fileName=name;
            fr = new FileReader(name);
            in = new BufferedReader(fr,MAXBUFFERSIZE);
            token = new StreamTokenizer(in);
            markerToken= new StreamTokenizer(in);
            token.wordChars(' ',' ');
            token.wordChars('_','_');
            token.whitespaceChars(',',',');
            token.slashStarComments(true);
            markPoint(); //Mark start of file for rewind
        }
        catch(Exception ex) {
                throw new RuntimeException("File "+name+" not found Exception in InFile constructor :"+ex.toString()+"   Current token is >"+token.sval);
        }
    }


    public InFile(String name, char sep){
        try{
            fileName=name;
            fr = new FileReader(name);
            in = new BufferedReader(fr,MAXBUFFERSIZE);
            token = new StreamTokenizer(in);
            token.whitespaceChars(sep,sep);
            token.ordinaryChar('_');
            token.slashStarComments(true);
            markPoint(); //Mark start of file for rewind
        } catch(Exception ex){
                throw new RuntimeException("File "+name+" with Separator "+sep+" not found Exception in InFile constructor :"+ex.toString()+"   Current token is >"+token.sval);
        }
    }
//Reopenfile
    public void reopen(){
        try{
            fr = new FileReader(fileName);
            in = new BufferedReader(fr,MAXBUFFERSIZE);
            token = new StreamTokenizer(in);
            markerToken= new StreamTokenizer(in);
            token.wordChars(' ',' ');
            token.wordChars('_','_');
            token.whitespaceChars(',',',');
            token.slashStarComments(true);
            markPoint(); //Mark start of file for rewind
        }
        catch(Exception ex){
                throw new RuntimeException("File "+fileName+"  not found on call to reopen() in InFile :"+ex.toString()+"   Current token is >"+token.sval);
        }
    }
    public String getName(){return fileName;}
    
//CSV file
    public void openFile(String name){
        try{
            fileName=name;
            fr = new FileReader(name);
            in = new BufferedReader(fr,MAXBUFFERSIZE);
            token = new StreamTokenizer(in);
            markerToken= new StreamTokenizer(in);
            token.wordChars(' ',' ');
            token.wordChars('_','_');
            token.whitespaceChars(',',',');
            token.slashStarComments(true);
        }
        catch(Exception ex){
                throw new RuntimeException("File "+name+"  not found on call to openFile() in InFile :"+ex.toString()+"   Current token is >"+token.sval);
        }
    }
	
//CSV file
    public void closeFile(){
        try {
            in.close();
            fr.close();
        } catch (IOException ex) {
                throw new RuntimeException("Failed to close  "+fileName+"  not found on call to closeFile() in InFile :"+ex.toString()+"   Current token is >"+token.sval);
        }
    }
    public char readChar(){
        char c;
        try{
            c=(char)in.read();
        }
        catch(Exception ex){
                throw new RuntimeException("Failed to read a character from  "+fileName+"  readChar() in InFile :"+ex.toString()+"Current token is >"+token.sval);
        }
        return(c);
    }
//Reads and returns
//Problems: Ignoring comments prior to the line
//Returns null if EOF??
    public String readLine() {
        String v=null;
        try{
                //To force ignore of comments preceeding the line
/*			//CHECK
                token.pushBack();
                token.nextToken();
*/			
            v=in.readLine();
        }
        catch(Exception ex) {
                throw new RuntimeException("Failed to read a line from  "+fileName+"  readLine() in InFile :"+ex.toString()+"Current token is >"+token.sval);
        }
        return(v);
    }

    public Object read(){
        int v=0;
        int t;
        Object o=null;
        try{
            t=token.nextToken();
            if(t==StreamTokenizer.TT_NUMBER)
                o= new Double(token.nval);
            else
                o=token.sval;
        }
        catch(IOException ex){
                throw new RuntimeException("Failed to read a line from  "+fileName+"  read() in InFile :"+ex.toString()+"Current token is >"+token.sval);
        }
        return o;
    }
    public int readInt(){
        int v=0;
        int t;
        try{
            t=token.nextToken();
            if(t!=StreamTokenizer.TT_NUMBER) {
                    System.out.println("ERROR: Attempting to read a non integer");
                    System.out.println("Current token is >"+token.sval);
                    System.out.println("File name ="+fileName);
                    throw new RuntimeException("Failed to read an integer from  "+fileName+"  readInt() in InFile Current token is >"+token.sval);
            }
            v= (int)token.nval;
        }
        catch(Exception ex){
                    throw new RuntimeException("Failed to read an integer from  "+fileName+"  readInt() in InFile Current token is >"+token.sval);
        }
        return(v);
    }

    public double readDouble(){
        double v=0;
        try{
            int t =token.nextToken();
            if(t!=StreamTokenizer.TT_NUMBER){
                    System.out.println("ERROR: Attempting to read a non double");
                    System.out.println("Current token is >"+token.sval);
                    System.out.println("File name ="+fileName);
                    throw new RuntimeException("Failed to read a double from  "+fileName+"  readDouble() in InFile Current token is >"+token.sval);
            }
            v= token.nval;
        }
        catch(Exception ex){
            throw new RuntimeException("Failed to read a double from  "+fileName+"  readDouble() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(v);
    }
    public float readFloat(){
        double v=0;
        try{
            int t =token.nextToken();
            if(t!=StreamTokenizer.TT_NUMBER){
                System.out.println("ERROR: Attempting to read a non double");
                System.out.println("Current token is >"+token.sval);
                System.out.println("File name ="+fileName);
                throw new RuntimeException("Failed to read a float from  "+fileName+"  readFloat() in InFile Current token is >"+token.sval);
            }
            v= token.nval;
        }
        catch(Exception ex) {
                System.out.println("ERROR: wrong Format");
                System.out.println("File name ="+fileName);
                throw new RuntimeException("Failed to read a float from  "+fileName+"  readFloat() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return((float)v);
    }
    public String readStringToChar(char delimit)   {
        String v="";
        char[] name = new char[1];
        try{

            name[0]=readChar();
            while(name[0]==' '||name[0]=='\n'|| name[0]=='\t')
                    name[0]=readChar();
            while(name[0]!=' ' && name[0]!='\n' && name[0]!='\t' && name[0]!=delimit) //name!=EOF && )
            {
                    v+=new String(name);
                    name[0]=readChar();
            }
            while(name[0]!=delimit) //name!=EOF && )
            {
                    name[0]=readChar();
            }


        }
        catch(Exception ex){
            throw new RuntimeException("Failed to read a straing from chars from  "+fileName+"  readStringToChar() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(v);
    }

    public String readStringIgnoreWhite() {
        String v="";
        char[] name = new char[1];
        try{
            name[0]=readChar();
            while(name[0]==' '||name[0]=='\n'|| name[0]=='\t')
                    name[0]=readChar();
            while(name[0]!=' ' && name[0]!='\n' && name[0]!='\t') //name!=EOF && )
            {
                    v+=new String(name);
                    name[0]=readChar();
            }
        }
        catch(NoSuchElementException ex)
        {
        throw new RuntimeException("Failed to read a string from chars from  "+fileName+"  readStringIgnoreWhite() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(v);
    }

    public String readString(){
        String v;
        int t;
        try{
            t=token.nextToken();
            if(t!=StreamTokenizer.TT_WORD)
            {
                    System.out.println("ERROR: Attempting to read a non string");
                    System.out.println("Current token is >"+token.sval+"\t t ="+token.nval+"\t"+token.toString());
                System.out.println("File name ="+fileName);
                    System.exit(0);
            }
            v= token.sval;
        }
        catch(IOException ex){
            throw new RuntimeException("Failed to read a string from  "+fileName+"  readString() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(v);
    }
//Reads header line delimited by:Vector! oollld code
    public Vector readStringLine(String delimit){
        Vector headers = new Vector();
        String line;
        String name;
        try{
            line=readLine();

            StringTokenizer sToke = new StringTokenizer(line,delimit);
            while(sToke.hasMoreTokens()) {
                    name=sToke.nextToken();
                    headers.addElement(name);
            }
        }
        catch(NoSuchElementException ex) {
            throw new RuntimeException("Failed to read a string from  "+fileName+"  readStringLine(String delimit) in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(headers);
    }
    public Vector readStringLine(){
        Vector headers = new Vector();
        String line;
        String name;
        try{
            line=readLine();
            StringTokenizer sToke = new StringTokenizer(line);
            while(sToke.hasMoreTokens())
            {
                    name=sToke.nextToken();
                    headers.addElement(name);
            }
        }
        catch(NoSuchElementException ex){
            throw new RuntimeException("Failed to read a string from  "+fileName+"  readStringLine() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(headers);
    }



    //PRE: EOF NOT reached during the line
    //POST: Returns NULL if first read is EOF
    public double[] readDoubleLine(int size){
        double d=readDouble();
        if(token.ttype==StreamTokenizer.TT_EOF)
                return(null);
        double[] data = new double[size];
        data[0]=d;
        for(int i=1;i<size;i++)
            data[i]=readDouble();
        return(data);
    }

    //POST: Returns FALSE if first read is EOF
    public boolean readDoubleLine(double[] data) {
        data[0]=this.readDouble();
        if(token.ttype==StreamTokenizer.TT_EOF)
            return(false);

        for(int i=1;i<data.length;i++)
             data[i]=this.readDouble();
        return(true);
    }

//Reads upto the first occurence of delimit string
//VERY INEFFICIENT AND HACKED
//String conversion bad
//Should check EOF
//Shouldnt use += for string
    public String readStringUpTo(char delimit)
    {
        char[] name = new char[1];
        String header="";
        try{
            name[0]=readChar();
            while(name[0]!=delimit) //name!=EOF && )
            {
                    header+=new String(name);
                    name[0]=readChar();
            }
        }
        catch(NoSuchElementException ex) {
            System.out.println("IO Exception caught in readStringUpTo");
            throw new RuntimeException("Failed to read a string from  "+fileName+"  readStringUpTo() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
        return(header);
    }


    //Sets a marker in the BufferedStream that should persist until the
    //file is finished
    public void markPoint(){
        try{
            in.mark(MAXBUFFERSIZE-1);
        }
        catch(IOException ex)    {
            throw new RuntimeException("Failed to mark a point  from  "+fileName+"  markPoint() in InFile Current token is >"+token.sval+" exception ="+ex);
        }

    }
    public void rewind(){
        try{
           in.reset();
        }
        catch(IOException ex){
            throw new RuntimeException("Failed to mark a point  from  "+fileName+"  rewind() in InFile Current token is >"+token.sval+" exception ="+ex);
        }
    }
    public int countLines(){
        int count =0;
        String str=readLine();
        while(str!=null){
                str=readLine();
                count++;
        }
//        rewind();
        return(count);
    }

    //Test Harness
    static public void main(String[] args)
    {


            InFile t=new InFile("C:/JavaSource/FileIO/test.csv");
            int a,b;
            double x;
            String s,s2;
            a=t.readInt();
            x=t.readDouble();
            s=t.readStringUpTo(',');
            System.out.println(s+"\tEND");
            StringTokenizer st = new StringTokenizer(s);
            s2=st.nextToken("-");
            System.out.println(s2+"STRING");
            a= Integer.parseInt(s2);
            System.out.println(a+"INTEGER");

//		a=(Integer.getInteger(s2)).intValue();
//		System.out.println(a+"\tEND");

    }
    public static boolean directoryExists(String s){
        File f= new File(s);
        if(f.exists() && f.isDirectory())
            return true;
        return false;
    }
    public static boolean fileExists(String s){
        File f= new File(s);
        if(f.exists() && !f.isDirectory())
            return true;
        return false;
    }
    public static boolean deleteFile(String s){
        File f= new File(s);
        if(f.exists()&& !f.isDirectory()){
            f.delete();
            return true;
        }
        return false;
    }     
}
