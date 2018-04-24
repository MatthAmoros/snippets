/****** Object:  StoredProcedure [dbo].[SEQ_NEXT_ID]    Script Date: 24/04/2018 16:00:00 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Matthieu
-- Create date: 28-02-2018
-- Description:	Devuelve un identificator unico formato GS1
-- =============================================
ALTER PROCEDURE [dbo].[SEQ_NEXT_ID] 
@PREFIJO NVARCHAR(8)
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

    --PREFIJO = [1][780][465287] => [DE 1 - 5][PREFIX PAIS][PREFIX EMPRESA]
	  SELECT  FORMAT((NEXT VALUE FOR dbo.CORRELATIVO_LOTES),CONCAT(@PREFIJO, '##########')) AS Id;  
END
