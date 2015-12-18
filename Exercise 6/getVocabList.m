function vocabList = getVocabList()
%GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
%cell array of the words
%   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
%   and returns a cell array of the words in vocabList.


%% Read the fixed vocabulary list
fid = fopen('vocab.txt');

% Store all dictionary words in cell array vocab{}
n = 1899;  % Total number of words in the dictionary
num = zeros(n,1);
% For ease of implementation, we use a struct to map the strings => integers
% In practice, you'll want to use some form of hashmap
vocab = cell(n, 1);
for i = 1:n
    % Word Index (can ignore since it will be = i)
    fscanf(fid, '%d', 1);
    % Actual Word
    vocab{i} = fscanf(fid, '%s', 1);
    num(i) = i;
end
vocabList = containers.Map(vocab,num);
fclose(fid);

end
