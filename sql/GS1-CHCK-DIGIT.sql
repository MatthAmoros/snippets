USE [erpFrusys]
GO
/****** Object:  UserDefinedFunction [dbo].[get_check_digit_gs1_sscc]    Script Date: 24/04/2018 16:40:03 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
 ALTER FUNCTION [dbo].[get_check_digit_gs1_sscc]
(
  @SSCC AS VARCHAR(17)
)
RETURNS INTEGER
AS BEGIN
declare @TOTAL_INTER int
SET @TOTAL_INTER = (3* CAST(SUBSTRING(@SSCC, 1, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 2, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 3, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 4, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 5, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 6, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 7, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 8, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 9, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 10, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 11, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 12, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 13, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 14, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 15, 1) AS INTEGER)
                + CAST(SUBSTRING(@SSCC, 16, 1) AS INTEGER)
                + 3* CAST(SUBSTRING(@SSCC, 17, 1) AS INTEGER)
         );
  RETURN (10 -(@TOTAL_INTER % 10) % 10)
END;
