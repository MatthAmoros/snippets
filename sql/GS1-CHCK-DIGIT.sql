 CREATE FUNCTION dbo.get_check_digit_gs1_sscc
(
  @SSCC AS VARCHAR(17)
)
RETURNS INTEGER
AS BEGIN
  RETURN (10 - (3* CAST(SUBSTRING('0' + @SSCC, 1, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 2, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 3, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 4, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 5, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 6, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 7, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 8, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 9, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 10, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 11, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 12, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 13, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 14, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 15, 1) AS INTEGER)
                + CAST(SUBSTRING('0' + @SSCC, 16, 1) AS INTEGER)
                + 3* CAST(SUBSTRING('0' + @SSCC, 17, 1) AS INTEGER)
         ) % 10)
END;
