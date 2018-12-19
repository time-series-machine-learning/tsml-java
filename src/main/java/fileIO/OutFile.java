package fileIO;
import java.io.*;

public class OutFile{

	private FileWriter fw;
	private BufferedWriter bw;
	protected PrintWriter outFile;
        protected String name;
	private char delimit;

	public OutFile(String name)
	{
            this.name=name;
            try
            {
                    fw = new FileWriter(name);
                    bw = new BufferedWriter(fw);
                    outFile = new PrintWriter(fw);
                    delimit=' ';
            }
            catch(IOException exception)
            {	
                    System.err.println(exception+" File "+ name+" Not found");
            }
	}
	public OutFile(String name, char delimiter)
	{
		try
		{
			fw = new FileWriter(name);
			bw = new BufferedWriter(fw);
			outFile = new PrintWriter(fw);
			delimit=delimiter;
		}
		catch(IOException exception)
		{	
			System.out.println(" File "+ name+" Not found");
		}
	}
        public OutFile(String name, boolean append)
	{
		try
		{
			fw = new FileWriter(name, append);
			bw = new BufferedWriter(fw);
			outFile = new PrintWriter(fw);
			delimit=' ';
		}
		catch(IOException exception)
		{	
			System.out.println(" File "+ name+" Not found");
		}
	}
//Reads and returns single line
	public boolean writeString(String v)
	{
			outFile.print(v);
			if(outFile.checkError())
				return(false);
			return(true);
	}

	public boolean writeLine(String v)
	{
			outFile.print(v+"\n");
			if(outFile.checkError())
				return(false);
			return(true);
	}
	public boolean writeInt(int v)
	{
		outFile.print(""+v+delimit);
		if(outFile.checkError())
			return(false);
		return(true);
	}
	public boolean writeChar(char c)
	{
		outFile.print(c);
			if(outFile.checkError())
				return(false);
			return(true);
	}
	public boolean writeBoolean(boolean b)
	{
		outFile.print(b);
			if(outFile.checkError())
				return(false);
			return(true);
	}
	public boolean writeDouble(double v)
	{
		outFile.print(""+v+delimit);
			if(outFile.checkError())
				return(false);
			return(true);
	}
	public boolean newLine()
	{
		outFile.print("\n");		
		if(outFile.checkError())
				return(false);
			return(true);
	}
	
	public void closeFile()
	{
		outFile.close();
	}
}
	