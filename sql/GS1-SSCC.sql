USE [TEST]
GO
/****** Object:  StoredProcedure [dbo].[SEQ_NEXT_ID]    Script Date: 24/04/2018 17:20:15 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Matthieu
-- Create date: 28-02-2018
-- Description:	Devuelve un identificator unico
-- =============================================
ALTER PROCEDURE [dbo].[SEQ_NEXT_ID] 
@PREFIJO NVARCHAR(10),
@SSCC_WHITHOUT_CD NVARCHAR(17) = NULL,
@SSCC_CD NVARCHAR(1) = NULL
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    -- Insert statements for procedure here
	  SET @SSCC_WHITHOUT_CD = (FORMAT((NEXT VALUE FOR dbo.CORRELATIVO_LOTES),CONCAT(@PREFIJO, '#######')));
	  set @SSCC_CD = (SELECT [dbo].[get_check_digit_gs1_sscc] (@SSCC_WHITHOUT_CD));
	  SELECT CONCAT(@SSCC_WHITHOUT_CD, @SSCC_CD);
END
