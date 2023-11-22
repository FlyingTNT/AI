package common.network.layers;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class DatasetLoader {
	
	static String PROBLEM_FILE_NAME = "problem_tokenized_";
	static String SUMBISSION_FILE_NAME = "submission_tokenized_";
	
	static float[][] loadProblems(String directoryPath)
	{
		return loadFilePattern(directoryPath, PROBLEM_FILE_NAME);
	}
	
	static float[][] loadSubmissions(String directoryPath)
	{
		return loadFilePattern(directoryPath, SUMBISSION_FILE_NAME);
	}
	
	static float[][][] loadSubmissions2d(String directoryPath)
	{
		return loadFilePatternInterleaved(directoryPath, SUMBISSION_FILE_NAME, 2);
	}
	
	static float[][] loadFilePattern(String directoryPath, String pattern)
	{
		File folder = new File(directoryPath);
		
		File[] problemFiles = folder.listFiles(new FileFilter() {
			
			@Override
			public boolean accept(File pathname) {
				if(pathname.getName().length() < pattern.length())
					return false;
				return pathname.getName().substring(0, pattern.length()).equals(pattern);
			}
		});
		
		float[][] out = new float[problemFiles.length][];
		
		int sequenceLength = -1;
		
		for(int i = 0; i < problemFiles.length; i++)
		{
			System.out.println("Loading: " + problemFiles[i].getName());
			int problemNumber = Integer.parseInt(problemFiles[i].getName().substring(pattern.length(),  problemFiles[i].getName().length() - 4));
			
			try {
				Scanner scanner = new Scanner(problemFiles[i]);
				if(sequenceLength == -1)
				{
					ArrayList<Float> tokenized = new ArrayList<>();
					while(scanner.hasNextInt())
					{
						tokenized.add((float)scanner.nextInt());
					}
					
					sequenceLength = tokenized.size();
					out[problemNumber] = new float[sequenceLength];
					for(int j = 0; j < sequenceLength; j++)
						out[problemNumber][j] = tokenized.get(j);
				}else {
					out[problemNumber] = new float[sequenceLength];
					for(int j = 0; j < sequenceLength; j++)
						out[problemNumber][j] = scanner.nextInt();
				}
				scanner.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		return out;
	}
	
	static float[][][] loadFilePatternInterleaved(String directoryPath, String pattern, int streams)
	{
		File folder = new File(directoryPath);
		
		File[] problemFiles = folder.listFiles(new FileFilter() {
			
			@Override
			public boolean accept(File pathname) {
				if(pathname.getName().length() < pattern.length())
					return false;
				return pathname.getName().substring(0, pattern.length()).equals(pattern);
			}
		});
		
		float[][][] out = new float[problemFiles.length][streams][];
		
		int sequenceLength = -1;
		
		for(int i = 0; i < problemFiles.length; i++)
		{
			System.out.println("Loading: " + problemFiles[i].getName());
			int problemNumber = Integer.parseInt(problemFiles[i].getName().substring(pattern.length(),  problemFiles[i].getName().length() - 4));
			
			try {
				Scanner scanner = new Scanner(problemFiles[i]);
				if(sequenceLength == -1)
				{
					ArrayList<Float> tokenized = new ArrayList<>();
					while(scanner.hasNextInt())
					{
						tokenized.add((float)scanner.nextInt());
					}
					
					sequenceLength = tokenized.size() / streams;
					out[problemNumber] = new float[streams][sequenceLength];
					for(int j = 0; j < sequenceLength; j++)
						for(int k = 0; k < streams; k++)
							out[problemNumber][k][j] = tokenized.get(j*streams+k);
				}else {
					out[problemNumber] = new float[streams][sequenceLength];
					for(int j = 0; j < sequenceLength; j++)
						for(int k = 0; k < streams; k++)
							out[problemNumber][k][j] = scanner.nextInt();
				}
				scanner.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		return out;
	}
}
