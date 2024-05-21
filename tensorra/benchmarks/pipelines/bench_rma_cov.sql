CREATE TABLE conference_A_100 AS 
SELECT conference_100.*
FROM conference_100 
INNER JOIN ranking ON conference_100.conference = ranking.conference
WHERE ranking.rating = 'A++';
CREATE TABLE conference_A_1k AS 
SELECT conference_1k.*
FROM conference_1k 
INNER JOIN ranking ON conference_1k.conference = ranking.conference
WHERE ranking.rating = 'A++';
CREATE TABLE conference_A_10k AS 
SELECT conference_10k.*
FROM conference_10k 
INNER JOIN ranking ON conference_10k.conference = ranking.conference
WHERE ranking.rating = 'A++';
CREATE TABLE conference_A_100k AS 
SELECT conference_100k.*
FROM conference_100k 
INNER JOIN ranking ON conference_100k.conference = ranking.conference
WHERE ranking.rating = 'A++';
CREATE TABLE conference_A_1m AS 
SELECT conference_1m.*
FROM conference_1m 
INNER JOIN ranking ON conference_1m.conference = ranking.conference
WHERE ranking.rating = 'A++';