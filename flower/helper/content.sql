CREATE TABLE content (company VARCHAR(100), 
					  name VARCHAR(100), 
					  prodLink VARCHAR(250),
					  imgLink VARCHAR(250), 
					  imgLoc VARCHAR(250), 
					  price DECIMAL(6,2),
					  UNIQUE KEY(company,name));
					  
CREATE TABLE contentBackup LIKE content; 
INSERT contentBackup SELECT * FROM content;